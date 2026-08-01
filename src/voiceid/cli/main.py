"""VoiceID CLI entrypoints."""

import argparse

import torch

from voiceid.features.log_mel import LogMelSpectrogram
from voiceid.inference.verifier import SpeakerVerifier
from voiceid.models.xvector import XVectorEncoder


def build_verifier(device: str = "cpu") -> SpeakerVerifier:
    extractor = LogMelSpectrogram()
    model = XVectorEncoder()
    return SpeakerVerifier(extractor, model, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(prog="voiceid")
    sub = parser.add_subparsers(dest="command", required=True)

    enroll_p = sub.add_parser("enroll")
    enroll_p.add_argument("--speaker-id", required=True)
    enroll_p.add_argument("--audio", required=True)

    verify_p = sub.add_parser("verify")
    verify_p.add_argument("--speaker-id", required=True)
    verify_p.add_argument("--audio", required=True)
    verify_p.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    verifier = build_verifier(device="cpu")

    if args.command == "enroll":
        verifier.enroll(args.speaker_id, args.audio)
        print(f"Enrolled {args.speaker_id}")
    elif args.command == "verify":
        score, decision = verifier.verify(args.speaker_id, args.audio, threshold=args.threshold)
        print(f"score={score:.4f} accept={decision}")
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
