import string
import easyocr
import pandas as pd
import numpy as np

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    if len(text) != 7:
        return False
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0','1','2','3','4','5','6','7','8','9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0','1','2','3','4','5','6','7','8','9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in range(7):
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break
    if foundIt:
        return vehicle_track_ids[car_indx]
    return -1, -1, -1, -1, -1


def interpolate_and_write_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    # Ensure frame numbers are numeric
    df['frame_nmr'] = pd.to_numeric(df['frame_nmr'], errors='coerce')
    df = df.dropna(subset=['frame_nmr'])
    df['frame_nmr'] = df['frame_nmr'].astype(int)

    interpolated_rows = []

    for car_id in df['car_id'].unique():
        car_df = df[df['car_id'] == car_id].sort_values('frame_nmr')

        # Create full frame range for the car
        all_frames = pd.DataFrame({'frame_nmr': range(car_df['frame_nmr'].min(), car_df['frame_nmr'].max() + 1)})
        car_df = pd.merge(all_frames, car_df, on='frame_nmr', how='left')

        # Interpolate numeric columns
        for col in ['license_plate_bbox_score', 'license_number_score']:
            if col in car_df.columns:
                car_df[col] = car_df[col].interpolate(method='linear', limit_direction='both')

        # Forward/backward fill for text/bbox columns
        for col in ['license_number', 'car_bbox', 'license_plate_bbox']:
            if col in car_df.columns:
                car_df[col] = car_df[col].ffill().bfill()

        car_df['car_id'] = car_id
        interpolated_rows.append(car_df)

    df_interpolated = pd.concat(interpolated_rows).sort_values(['frame_nmr', 'car_id'])

    # Overwrite output CSV
    df_interpolated.to_csv(output_csv_path, index=False)
