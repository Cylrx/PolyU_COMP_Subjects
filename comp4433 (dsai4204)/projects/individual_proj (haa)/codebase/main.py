from src.classify import main as classify_main
from src.cluster import main as clustering_main
from src.mining import main as mining_main
from src.eda import main as eda_main

def main():
    print("Choose your option: ")
    print("\t1. Run EDA")
    print("\t2. Run Classification")
    print("\t3. Run Clustering")
    print("\t4. Run Rule Mining")
    choice = input("Enter your choice: ")

    match choice:
        case "1": eda_main()
        case "2": classify_main()
        case "3": clustering_main()
        case "4": mining_main()
        case _: print("Invalid choice")

if __name__ == "__main__":
    main()
