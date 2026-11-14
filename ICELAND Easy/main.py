"""Clean a legacy geothermal station export."""

import csv
from decimal import Decimal, ROUND_HALF_EVEN


INPUT_COLUMNS = (
    "reading_id",
    "station_name",
    "temperature_raw",
    "unit",
    "reading_date",
)
OUTPUT_COLUMNS = ("reading_id", "station_name", "temperature_c", "reading_date")


def restore_name(name: str) -> str:
    """Undo UTF-8 text that was mistakenly decoded as a Western encoding."""
    for _ in range(3):
        if not any(marker in name for marker in ("Ã", "Â", "â")):
            break
        for encoding in ("cp1252", "latin-1"):
            try:
                repaired = name.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            name = repaired
            break
        else:
            break
    return name


def temperature_in_celsius(raw: str, unit: str) -> str:
    value = Decimal(raw.strip().replace(",", "."))
    if unit.strip().upper() == "F":
        value = (value - Decimal(32)) * Decimal(5) / Decimal(9)
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_EVEN))


def main() -> None:
    readings: dict[int, tuple[str, str, str]] = {}

    with open("test-input.csv", encoding="utf-8-sig", newline="") as source:
        for line in source:
            if tuple(part.strip() for part in line.strip().split(";")) == INPUT_COLUMNS:
                break
        else:
            raise ValueError("Input header not found")

        reader = csv.DictReader(source, fieldnames=INPUT_COLUMNS, delimiter=";")
        for row in reader:
            if not row["reading_id"] or row["reading_id"].lstrip().startswith("#"):
                continue
            reading_id = int(row["reading_id"].strip())
            if reading_id not in readings:
                readings[reading_id] = (
                    restore_name(row["station_name"].strip()),
                    temperature_in_celsius(row["temperature_raw"], row["unit"]),
                    row["reading_date"],
                )

    with open("test-output.csv", "w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n", quoting=csv.QUOTE_NONE)
        writer.writerow(OUTPUT_COLUMNS)
        for reading_id in sorted(readings):
            writer.writerow((reading_id, *readings[reading_id]))


if __name__ == "__main__":
    main()
