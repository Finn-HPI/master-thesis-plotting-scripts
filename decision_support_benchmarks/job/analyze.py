import argparse
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to directory containing *-PQP.txt files",
    )
    args = parser.parse_args()

    # Find all *-PQP.txt files
    txt_files = [f for f in os.listdir(args.dir) if f.endswith("-PQP.txt")]

    # Group files by prefix before first "-"
    prefix_map = {}
    for filename in sorted(txt_files):  # sort to ensure consistent "first file"
        prefix = filename.split("-")[0]
        if prefix not in prefix_map:
            prefix_map[prefix] = os.path.join(args.dir, filename)

    # Regex to match: Radix bits: <digit>
    pattern = re.compile(r"Radix bits: (\d)")
    pattern_left = re.compile(r"left_rows: (\d+),")
    pattern_right = re.compile(r"right_rows: (\d+),")

    # Process each selected file
    for prefix, filepath in prefix_map.items():
        with open(filepath, "r") as f:
            print(prefix, ": ", end="")
            grater_one_million = False
            big_sides = []
            for line in f:
                match = pattern.search(line)
                match_left = pattern_left.search(line)
                match_right = pattern_right.search(line)
                if match and match_left and match_right:
                    radix_bits = int(match.group(1))
                    left = int(match_left.group(1))
                    right = int(match_right.group(1))
                    min_rows = min(left, right)
                    if min_rows > 1000000:
                        grater_one_million = True
                        big_sides.append((min_rows / 1000000, radix_bits))
                    print(f"({radix_bits}, {min_rows}), ", end="")
            print("")
            if grater_one_million:
                print("==> > 1M", big_sides)


if __name__ == "__main__":
    main()
