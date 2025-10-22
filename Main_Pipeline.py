from Classic_Face_Detect import main as classic
from Feat_Track import main as features
from BgFg_Seg import main as segment

def main():
    while True:
        print("\n=== Face Detection Pipeline")
        print("B  -  Background/Foreground Segmentation")
        print("D  -  Classic Face Detection")
        print("T  -  Feature Track")
        print("E  -  Exit")

        choice = input("Enter: ").strip().upper()

        if choice == 'B':
            print("\nRunning Background/Foreground Segmentation...")
            segment()
        elif choice == 'D':
            print("\nRunning Classic Face Detection...")
            classic()
        elif choice == 'T':
            print("\nRunning Feature Track...")
            features()
        elif choice == 'E':
            print("\nExiting...")
            break
        else:
            print("Invalid option. Please enter new option.")
        
if __name__ == "__main__":
    main()