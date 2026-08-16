from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hphbiome.knowledge import CuratedKnowledgeRecord
from hphbiome.reference import ScientificReference


def _normalize_records(value: object) -> tuple[CuratedKnowledgeRecord, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError(
            'records must be an ordered sequence of CuratedKnowledgeRecord'
        )

    records: list[CuratedKnowledgeRecord] = []
    for index, record in enumerate(value):
        if not isinstance(record, CuratedKnowledgeRecord):
            raise TypeError(
                f'records[{index}] must be a CuratedKnowledgeRecord'
            )
        records.append(record)
    return tuple(records)


def _normalize_references(value: object) -> tuple[ScientificReference, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError(
            'references must be an ordered sequence of ScientificReference'
        )

    references: list[ScientificReference] = []
    for index, reference in enumerate(value):
        if not isinstance(reference, ScientificReference):
            raise TypeError(
                f'references[{index}] must be a ScientificReference'
            )
        references.append(reference)
    return tuple(references)


def _validate_unique_record_ids(
    records: tuple[CuratedKnowledgeRecord, ...],
) -> None:
    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        previous = seen.get(record.id)
        if previous is not None:
            raise ValueError(
                f'records[{index}].id duplicates records[{previous}].id: '
                f'{record.id!r}'
            )
        seen[record.id] = index


def _validate_unique_reference_aliases(
    references: tuple[ScientificReference, ...],
) -> None:
    identifiers: dict[str, int] = {}
    canonical_urls: dict[str, int] = {}

    for index, reference in enumerate(references):
        if reference.identifier is not None:
            previous = identifiers.get(reference.identifier)
            if previous is not None:
                raise ValueError(
                    f'references[{index}].identifier duplicates '
                    f'references[{previous}].identifier: '
                    f'{reference.identifier!r}'
                )
            identifiers[reference.identifier] = index

        if reference.canonical_url is not None:
            previous = canonical_urls.get(reference.canonical_url)
            if previous is not None:
                raise ValueError(
                    f'references[{index}].canonical_url duplicates '
                    f'references[{previous}].canonical_url: '
                    f'{reference.canonical_url!r}'
                )
            canonical_urls[reference.canonical_url] = index


@dataclass(frozen=True)
class KnowledgeCollection:
    """An immutable ordered collection of curated knowledge and references."""

    records: Sequence[CuratedKnowledgeRecord] = ()
    references: Sequence[ScientificReference] = ()

    def __post_init__(self) -> None:
        records = _normalize_records(self.records)
        references = _normalize_references(self.references)

        _validate_unique_record_ids(records)
        _validate_unique_reference_aliases(references)

        object.__setattr__(self, 'records', records)
        object.__setattr__(self, 'references', references)

    def get_record(self, record_id: str) -> CuratedKnowledgeRecord | None:
        """Return the record with ``record_id``, or ``None`` when missing."""
        return next(
            (record for record in self.records if record.id == record_id),
            None,
        )

    def get_reference_by_identifier(
        self, identifier: str
    ) -> ScientificReference | None:
        """Return the reference with ``identifier``, or ``None``."""
        return next(
            (
                reference
                for reference in self.references
                if reference.identifier is not None
                and reference.identifier == identifier
            ),
            None,
        )

    def get_reference_by_canonical_url(
        self, canonical_url: str
    ) -> ScientificReference | None:
        """Return the reference with ``canonical_url``, or ``None``."""
        return next(
            (
                reference
                for reference in self.references
                if reference.canonical_url is not None
                and reference.canonical_url == canonical_url
            ),
            None,
        )
