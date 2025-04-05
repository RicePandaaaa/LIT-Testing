import multilateration

def main():
    # Create an instance of the multilateration class.
    multilaterator = multilateration.Multilateration(simulate_tower_down=False, resolution=1.0)
    
    # Generate valid readings ensuring that all circles have a common intersection.
    # multilaterator.generate_random_readings()
    multilaterator.scanner.get_tower_signal()
    
    # multilaterate the position based on the available (or down-adjusted) towers.
    multilaterator.multilaterate()
    
    # Plot the towers (with circles) and the estimated position.
    multilaterator.run_animation()

    multilaterator.save_animation(filename="multilateration_animation.gif", fps=60)

if __name__ == "__main__":
    main()
