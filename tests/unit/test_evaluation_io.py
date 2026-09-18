from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import mido
import pytest
from piano_transcriber.cli import main
from piano_transcriber.evaluation.io import load_score
from piano_transcriber.evaluation.report import compare
from piano_transcriber.evaluation.score import evaluate
from piano_transcriber.evaluation.suite import evaluate_suite
from piano_transcriber.evaluation.synthetic import synthetic_corpus, write_corpus
from piano_transcriber.evaluation.types import EvaluationConfig


def xml_file(
    tmp_path: Path,
    measures: str,
    attributes: str = (
        "<divisions>12</divisions><time><beats>4</beats><beat-type>4</beat-type></time>"
    ),
) -> Path:
    path = tmp_path / "reference.musicxml"
    path.write_text(
        '<score-partwise version="4.0"><part-list/><part id="P1"><measure number="1">'
        f"<attributes>{attributes}</attributes>{measures}</measure></part></score-partwise>",
        encoding="utf-8",
    )
    return path


def xml_note(step: str = "C", duration: int = 12, extras: str = "") -> str:
    return (
        f"<note><pitch><step>{step}</step><octave>4</octave></pitch>"
        f"<duration>{duration}</duration>{extras}</note>"
    )


def test_musicxml_chord_backup_forward_staff_voice_and_missing_tempo(tmp_path: Path) -> None:
    path = xml_file(
        tmp_path,
        xml_note(extras="<voice>1</voice><staff>1</staff>")
        + xml_note("E", extras="<chord/><voice>1</voice><staff>1</staff>")
        + "<backup><duration>12</duration></backup><forward><duration>6</duration></forward>"
        + xml_note("G", 6, "<voice>2</voice><staff>2</staff>"),
    )
    score = load_score(path)
    assert [n.onset_beats for n in score.notes] == [0, 0, 0.5]
    assert score.notes[2].staff == "2"
    assert score.notes[2].voice == "2"
    assert all(n.onset_seconds is None for n in score.notes)
    report = evaluate(score, score)
    assert report["metrics"]["notes"]["f1"] == 1
    assert report["metrics"]["transcription"] is None


def test_musicxml_ties_are_merged_and_counted(tmp_path: Path) -> None:
    path = xml_file(
        tmp_path,
        xml_note(duration=48, extras='<tie type="start"/><voice>1</voice>')
        + '</measure><measure number="2">'
        + xml_note(duration=12, extras='<tie type="stop"/><voice>1</voice>'),
    )
    score = load_score(path)
    assert len(score.notes) == 1
    assert score.notes[0].duration_beats == 5
    assert score.structure["tie_count"] == 1
    assert score.structure["measure_count"] == 2


def test_musicxml_pickup(tmp_path: Path) -> None:
    path = xml_file(tmp_path, xml_note() + '</measure><measure number="1">' + xml_note(duration=48))
    path.write_text(
        path.read_text().replace('<measure number="1">', '<measure number="0" implicit="yes">', 1)
    )
    score = load_score(path)
    assert score.pickup_beats == 1
    assert score.first_downbeat_beats == 1
    assert score.measure_boundaries == (0, 1)
    assert score.downbeats == (1,)


def test_musicxml_tempo_change_and_rhythm_no_enharmonic_penalty(tmp_path: Path) -> None:
    events = (
        '<direction><sound tempo="60"/></direction>'
        + xml_note(duration=12)
        + '<direction><sound tempo="120"/></direction>'
        + xml_note("D", duration=12)
    )
    path = xml_file(tmp_path, events)
    score = load_score(path)
    assert [(n.onset_seconds, n.duration_seconds) for n in score.notes] == [(0, 1), (1, 0.5)]
    path.write_text(path.read_text().replace("<step>D</step>", "<step>C</step><alter>2</alter>"))
    alternate = load_score(path)
    result = evaluate(alternate, score)["metrics"]
    assert result["rhythm"]["rhythmic_value_accuracy"] == 1
    assert result["notes"]["f1"] == 1


def test_musicxml_missing_staff_voice_stay_missing(tmp_path: Path) -> None:
    reference = load_score(xml_file(tmp_path, xml_note()))
    report = evaluate(reference, reference)
    assert report["metrics"]["staff"]["accuracy"] is None
    assert report["metrics"]["voice"]["assignment_accuracy"] is None


def test_voice_scope_is_explicit_and_part_default_preserves_cross_staff_voice(
    tmp_path: Path,
) -> None:
    path = xml_file(
        tmp_path,
        xml_note(extras="<staff>1</staff><voice>1</voice>")
        + xml_note("E", extras="<staff>2</staff><voice>1</voice>"),
    )
    assert {n.voice for n in load_score(path).notes} == {"1"}
    assert {n.voice for n in load_score(path, voice_scope="staff").notes} == {"1:1", "2:1"}


def test_midi_tempo_changes_duplicate_pitch_and_no_staff_inference(tmp_path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend(
        [
            mido.MetaMessage("time_signature", numerator=3, denominator=4),
            mido.MetaMessage("set_tempo", tempo=1_000_000),
            mido.Message("note_on", note=60, velocity=80),
            mido.Message("note_on", note=60, velocity=80, time=240),
            mido.MetaMessage("set_tempo", tempo=500_000, time=240),
            mido.Message("note_off", note=60, time=0),
            mido.Message("note_on", note=60, velocity=0, time=480),
        ]
    )
    path = tmp_path / "reference.mid"
    midi.save(path)
    score = load_score(path)
    assert [(n.onset_beats, n.duration_beats) for n in score.notes] == [(0, 1), (0.5, 1.5)]
    assert [(n.onset_seconds, n.duration_seconds) for n in score.notes] == [(0, 1), (0.5, 1)]
    assert score.notes[0].measure == "1"
    assert score.notes[1].beat == 0.5
    assert score.time_signatures == ((0, 3, 4),)
    assert score.pickup_beats is None
    assert all(n.staff is None and n.voice is None and n.hand is None for n in score.notes)


def test_midi_no_explicit_tempo_uses_format_default(tmp_path: Path) -> None:
    midi = mido.MidiFile()
    midi.tracks.append(
        mido.MidiTrack(
            [mido.Message("note_on", note=60), mido.Message("note_off", note=60, time=480)]
        )
    )
    path = tmp_path / "default.mid"
    midi.save(path)
    score = load_score(path)
    assert score.notes[0].duration_seconds == 0.5
    assert score.time_signatures == ()
    assert any("default" in x for x in score.warnings)


def test_compressed_musicxml(tmp_path: Path) -> None:
    source = xml_file(tmp_path, xml_note())
    path = tmp_path / "reference.mxl"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="score.xml"/></rootfiles></container>',
        )
        archive.writestr("score.xml", source.read_text())
    assert load_score(path).notes == load_score(source).notes


def test_invalid_musicxml_duration_and_noncontiguous_tie(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-positive"):
        load_score(xml_file(tmp_path, xml_note(duration=0)))
    with pytest.raises(ValueError, match="noncontiguous"):
        load_score(
            xml_file(
                tmp_path,
                xml_note(extras='<tie type="start"/>')
                + "<forward><duration>12</duration></forward>"
                + xml_note(extras='<tie type="stop"/>'),
            )
        )


def test_raw_json_loading(tmp_path: Path) -> None:
    path = tmp_path / "transcription.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "test",
                "notes": [{"pitch": 60, "onset_seconds": 0.1, "offset_seconds": 0.6}],
            }
        )
    )
    score = load_score(path)
    assert score.stage == "transcription"
    assert score.notes[0].duration_seconds == 0.5
    assert score.notes[0].onset_beats is None


def test_cli_single_suite_compare_and_output_tables(tmp_path: Path) -> None:
    manifest = write_corpus(tmp_path / "corpus")
    output = tmp_path / "report.json"
    table = tmp_path / "report.tsv"
    prediction = manifest.parent / "scale.json"
    assert (
        main(
            [
                "evaluate",
                "--prediction",
                str(prediction),
                "--reference",
                str(prediction),
                "--json",
                str(output),
                "--table",
                str(table),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["metrics"]["notes"]["f1"] == 1
    assert table.read_text().startswith("piece\tmetric\tvalue")
    assert (
        main(
            [
                "compare-evaluations",
                str(output),
                str(output),
                "--json",
                str(tmp_path / "compare.json"),
            ]
        )
        == 0
    )
    assert main(["evaluate-suite", str(manifest), "--json", str(output)]) == 0
    suite = json.loads(output.read_text())
    assert suite["completed"] == len(synthetic_corpus())
    assert not suite["pending"]
    assert not suite["errors"]
    assert all(item["report"]["metrics"]["notes"]["f1"] == 1 for item in suite["pieces"])
    assert compare(suite, suite)["kind"] == "suite-comparison"


def test_yaml_relative_paths_pending_and_errors_are_explicit(tmp_path: Path) -> None:
    manifest = write_corpus(tmp_path / "corpus")
    yaml_path = tmp_path / "evaluation.yaml"
    yaml_path.write_text(
        "pieces:\n  - id: ready\n    prediction: corpus/scale.json\n"
        "    reference: corpus/scale.json\n  - id: needs-reference\n"
        "    audio: test.wav\n    reference: null\n  - id: needs-output\n"
        "    audio: test.wav\n    reference: corpus/scale.json\n"
        "  - id: absent-file\n    prediction: missing.mid\n    reference: corpus/scale.json\n"
    )
    result = evaluate_suite(yaml_path, EvaluationConfig())
    assert manifest.exists()
    assert result["completed"] == 1
    assert len(result["pending"]) == 2
    assert len(result["errors"]) == 1
    assert main(["evaluate-suite", str(yaml_path), "--json", str(tmp_path / "pending.json")]) == 2


def test_duplicate_manifest_ids_rejected(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    path.write_text('{"pieces": [{"id": "same"}, {"id": "same"}]}')
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_suite(path, EvaluationConfig())


def test_musicxml_meter_changes_beams_rests_and_tuplets(tmp_path: Path) -> None:
    path = xml_file(
        tmp_path,
        xml_note(duration=6, extras='<beam number="1">begin</beam>')
        + xml_note("E", 6, '<beam number="1">end</beam>')
        + "<note><rest/><duration>36</duration></note>"
        + '</measure><measure number="2"><attributes><time><beats>6</beats>'
        "<beat-type>8</beat-type></time></attributes>"
        + xml_note("G", 4, '<notations><tuplet type="start"/></notations>')
        + xml_note("A", 4)
        + xml_note("B", 4, '<notations><tuplet type="stop"/></notations>'),
    )
    score = load_score(path)
    assert score.time_signatures == ((0, 4, 4), (4, 6, 8))
    assert score.structure["rest_count"] == 1
    assert score.structure["beam_groups"] == 1
    assert score.structure["tuplet_groups"] == 1
    assert score.notes[-1].duration_beats == pytest.approx(1 / 3)


def test_explicit_clefs_provide_treble_bass_confusion(tmp_path: Path) -> None:
    path = xml_file(
        tmp_path,
        xml_note(extras="<staff>1</staff>"),
        "<divisions>12</divisions><staves>2</staves>"
        '<clef number="1"><sign>G</sign><line>2</line></clef>',
    )
    score = load_score(path)
    result = evaluate(score, score)["metrics"]["staff"]
    assert result["treble_bass_confusion"] == {"treble": {"treble": 1}}
    assert score.structure["staff_count"] == 2


def test_pipeline_musicxml_roundtrip_with_ties(tmp_path: Path) -> None:
    from piano_transcriber.evaluation.io import from_reconstructed, from_transcription
    from piano_transcriber.notation.musicxml import write_score_musicxml
    from piano_transcriber.score.reconstruct import ReconstructionConfig, reconstruct_score
    from piano_transcriber.transcription.types import NoteEvent, TranscriptionResult

    raw = TranscriptionResult((NoteEvent(0, 60, 5), NoteEvent(5, 64, 6)), "test", 6)
    reconstructed = reconstruct_score(raw, ReconstructionConfig(bpm=60))
    path = write_score_musicxml(reconstructed, tmp_path / "output.musicxml")
    loaded = load_score(path, voice_scope="staff")
    report = evaluate(loaded, from_reconstructed(reconstructed), EvaluationConfig(alignment="none"))
    assert report["metrics"]["notes"]["f1"] == 1
    assert report["metrics"]["rhythm"]["written_duration_beat_error"]["mae"] == 0
    assert report["metrics"]["voice"]["assignment_accuracy"] == 1
    assert from_transcription(raw).stage == "transcription"


def test_diagnostic_reader_preserves_hypotheses_and_explicit_hands(tmp_path: Path) -> None:
    path = tmp_path / "diagnostics.json"
    path.write_text(
        json.dumps(
            {
                "time_signature": "4/4",
                "bpm": 120,
                "measure_count": 1,
                "pickup_beats": "0",
                "first_full_downbeat_beats": "0",
                "meter_inference": {"hypotheses": [{"time_signature": "6/8"}]},
                "events": [
                    {
                        "pitch": 60,
                        "raw_onset_seconds": 0,
                        "raw_offset_seconds": 0.5,
                        "quantized_onset_beats": "0",
                        "written_duration_beats": "1",
                        "assigned_staff": 1,
                        "assigned_voice": 2,
                        "assigned_hand": "right",
                    }
                ],
            }
        )
    )
    score = load_score(path)
    assert score.notes[0].voice == "1:2"
    assert score.notes[0].hand == "right"
    assert score.meter_hypotheses == ({"time_signature": "6/8"},)


def test_cli_missing_input_returns_concise_failure(tmp_path: Path) -> None:
    assert (
        main(
            [
                "evaluate",
                "--prediction",
                str(tmp_path / "absent.mid"),
                "--reference",
                str(tmp_path / "absent.mid"),
                "--json",
                str(tmp_path / "metrics.json"),
            ]
        )
        == 2
    )
