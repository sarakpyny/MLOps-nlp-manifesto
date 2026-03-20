"""Build one merged dataset from local raw manifesto files."""

from pathlib import Path

from src.data.load_data import load_metadata, merge_metadata_with_texts


def main() -> None:
    """Build merged CSV and Parquet datasets from local raw sources."""
    metadata_path = Path("data/archelect_search.csv")
    text_files_path = Path("data/text_files")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "manifestos_raw.csv"
    output_parquet = output_dir / "manifestos_raw.parquet"

    print("Loading local metadata...")
    metadata = load_metadata(metadata_path)

    print("Merging local metadata with local text files...")
    df = merge_metadata_with_texts(metadata, text_files_path)

    print(f"Saving CSV to {output_csv} ...")
    df.to_csv(output_csv, index=False)

    print(f"Saving Parquet to {output_parquet} ...")
    df.to_parquet(output_parquet, index=False)

    print("Done.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
