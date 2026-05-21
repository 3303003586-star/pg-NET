import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm


def generate_batch_radio_maps():
    """Batch generate 56080 radio maps"""

    # System parameters
    tx_power = 23  # dBm
    frequency = 5.9e9  # Hz
    tx_height = 10  # Base station height
    rx_height = 1.5  # Receiver height

    # WinProp mapping range
    winprop_min_dbm = -120
    winprop_max_dbm = -40

    def dbm_to_pixel(dbm_value):
        """Convert dBm to pixel value"""
        pixel = (dbm_value - winprop_min_dbm) / (winprop_max_dbm - winprop_min_dbm) * 255
        return np.clip(pixel, 0, 255)

    def calculate_3gpp_pathloss(distance_2d, is_los, breakpoint_distance):
        """Calculate 3GPP path loss (no shadowing)"""
        fc_ghz = frequency / 1e9
        distance_3d = np.sqrt(distance_2d ** 2 + (tx_height - rx_height) ** 2)

        if distance_2d <= 0:
            return 0

        if is_los:
            if distance_2d <= breakpoint_distance:
                pl = 32.4 + 21 * np.log10(distance_3d) + 20 * np.log10(fc_ghz)
            else:
                pl = 32.4 + 40 * np.log10(distance_3d) + 20 * np.log10(fc_ghz) - 9.5 * np.log10(
                    (breakpoint_distance) ** 2 + (tx_height - rx_height) ** 2)
        else:
            pl = 22.4 + 35.3 * np.log10(distance_3d) + 21.3 * np.log10(fc_ghz) - 0.3 * (rx_height - 1.5)

        return pl

    def find_transmitter_position(tx_image):
        """Find transmitter position from image"""
        bright_spots = np.argwhere(tx_image > 200)
        if len(bright_spots) == 0:
            max_val = np.max(tx_image)
            bright_spots = np.argwhere(tx_image == max_val)

        if len(bright_spots) == 0:
            raise ValueError("No transmitter found in image")

        y, x = bright_spots[0]
        return x, y

    def get_directional_breakpoint(tx_position, los_map, direction_angle):
        """Get LOS boundary distance for specific direction"""
        tx_x, tx_y = tx_position
        map_size = los_map.shape[0]

        for distance in range(1, map_size):
            x = int(tx_x + distance * np.cos(direction_angle))
            y = int(tx_y + distance * np.sin(direction_angle))

            if x < 0 or x >= map_size or y < 0 or y >= map_size:
                return distance - 1

            if not los_map[y, x]:
                for d in range(distance, 0, -1):
                    x_prev = int(tx_x + d * np.cos(direction_angle))
                    y_prev = int(tx_y + d * np.sin(direction_angle))
                    if los_map[y_prev, x_prev]:
                        return d
                return 0

        return map_size - 1

    # Configure paths
    antennas_folder = r"E:\edge load\png\antennas"
    los_folder = r"E:\data\los+nlos"
    output_folder = r"E:\data\result"

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    print("Starting batch processing of 56080 radio maps...")
    print(f"Input antennas: {antennas_folder}")
    print(f"Input LOS maps: {los_folder}")
    print(f"Output folder: {output_folder}")

    # Process all 701 scenes, 80 transmitters each
    total_maps = 701 * 80
    processed_count = 0
    failed_count = 0

    for scene_id in tqdm(range(701), desc="Processing scenes"):
        for tx_id in range(80):
            try:
                # Load transmitter position image
                tx_path = os.path.join(antennas_folder, f"{scene_id}_{tx_id}.png")
                tx_img = np.array(Image.open(tx_path).convert('L'))
                tx_x, tx_y = find_transmitter_position(tx_img)

                # Load LOS/NLOS map
                los_path = os.path.join(los_folder, f"{scene_id}_{tx_id}_los.png")
                los_map = np.array(Image.open(los_path).convert('L')) > 128

                # Map size
                map_size = 256
                radio_map = np.zeros((map_size, map_size), dtype=np.uint8)

                # Precompute directional breakpoints
                num_directions = 720
                directional_breakpoints = {}

                for i in range(num_directions):
                    angle = 2 * np.pi * i / num_directions
                    breakpoint_dist = get_directional_breakpoint((tx_x, tx_y), los_map, angle)
                    directional_breakpoints[i] = breakpoint_dist

                # Generate radio map
                for y in range(map_size):
                    for x in range(map_size):
                        distance = np.sqrt((x - tx_x) ** 2 + (y - tx_y) ** 2)

                        if distance < 1:
                            radio_map[y, x] = dbm_to_pixel(tx_power)
                        else:
                            is_los = los_map[y, x]

                            dx = x - tx_x
                            dy = y - tx_y
                            direction_angle = np.arctan2(dy, dx)
                            direction_index = int((direction_angle % (2 * np.pi)) / (2 * np.pi) * num_directions)
                            breakpoint_distance = directional_breakpoints[direction_index]

                            pl_theoretical = calculate_3gpp_pathloss(distance, is_los, breakpoint_distance)
                            rx_power = tx_power - pl_theoretical
                            radio_map[y, x] = dbm_to_pixel(rx_power)

                # Save only the radio map (second image)
                output_path = os.path.join(output_folder, f"{scene_id}_{tx_id}.png")
                Image.fromarray(radio_map).save(output_path)

                processed_count += 1

            except Exception as e:
                failed_count += 1
                continue

    print(f"\nBatch processing completed!")
    print(f"Successfully processed: {processed_count}/{total_maps}")
    print(f"Failed: {failed_count}/{total_maps}")
    print(f"Output saved to: {output_folder}")


if __name__ == "__main__":
    generate_batch_radio_maps()