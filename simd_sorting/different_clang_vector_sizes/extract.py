import sys
import os

def parse_file(filename):
    num_tuples = None
    total_time_usecs = None

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("NUM-TUPLES:"):
                num_tuples = int(line.split(":")[1].strip())
            elif line.startswith("TOTAL-TIME-USECS:"):
                total_time_usecs = int(line.split(":")[1].strip())

    return num_tuples, total_time_usecs

def process_prefix(prefix, scale):
    num_tuples_list = []
    total_time_usecs_list = []
    
    for i in range(1, 6):
        filename = f"{prefix}_run{i}.txt"
        if os.path.exists(filename):
            num_tuples, total_time_usecs = parse_file(filename)
            num_tuples_list.append(num_tuples)
            total_time_usecs_list.append(total_time_usecs)
        else:
            print(f"Warning: {filename} not found!")

    avg_num_tuples = sum(num_tuples_list) / len(num_tuples_list)
    avg_total_time_usecs = sum(total_time_usecs_list) / len(total_time_usecs_list)

    print(f"{scale},{int(avg_num_tuples)},{int(avg_total_time_usecs)}")

# Used to construct averaged result csv files
# Example Usgae: python3 ../extract.py vec2_sort_
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename_prefix>")
        sys.exit(1)

    prefix = sys.argv[1]
    print("scale,num_tuples,time_us")
    for i in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        process_prefix(prefix + str(i), i)