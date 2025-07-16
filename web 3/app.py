from flask import Flask, render_template, request, jsonify
import overpy
import math
import requests
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import csv
from io import StringIO
import time
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
import re
from PIL import Image
import os
from datetime import datetime

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the Keras model and scalers
MODEL_PATH = 'solar_model.keras'


# Initialize model variables
model = None
#feature_scaler = None
#target_scaler = None
#feature_names = []
window_size = 24  # Default value if loading fails

try:
    # Load model and scalers
    model = load_model(MODEL_PATH)

    window_size = 24
    
    logger.info(f"Successfully loaded model with:")
    logger.info(f"Window size: {window_size}")

except Exception as e:
    logger.error(f"Error loading model or scalers: {str(e)}")



overpass_api = overpy.Overpass()

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--disable-notifications")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)


def create_output_directories():
    """Create directories for screenshots and analysis if they don't exist"""
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("shadow_analysis", exist_ok=True)
    print("Created output directories: 'screenshots' and 'shadow_analysis'")

def close_popups(driver):
    try:
        ok_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'OK') and contains(@style, 'background-color: rgb(255, 107, 0)')]"))
        )
        ok_button.click()
        print("Orange OK button clicked")
    except:
        try:
            close_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 
                    "button[aria-label='Close'], " +
                    ".close-button, " +
                    "[title='Close']"))
            )
            close_button.click()
            print("Generic close button clicked")
        except:
            print("No popup found")
            pass

def set_time_to_range(target_start, target_end,driver):
    try:
        time_label = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "label[style*='background: rgb(255, 107, 0)']"))
        )
        
        slider_handle = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[id='slider'] div[style*='cursor: grab']"))
        )
        
        slider_width = slider_handle.size['width']
        step_size = slider_width / 24
        
        actions = ActionChains(driver)
        
        max_attempts = 50
        current_attempt = 0
        last_time = None
        stuck_count = 0
        
        while current_attempt < max_attempts:
            current_time_text = time_label.text.strip()
            
            if any(month in current_time_text for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                print("Skipping date display, waiting for time...")
                time.sleep(1)
                current_attempt += 1
                continue
                
            print(f"Current: {current_time_text}, Target Range: {target_start}-{target_end}")
            
            if current_time_text == last_time:
                stuck_count += 1
                if stuck_count > 3:
                    print(f"Stuck at boundary: {current_time_text}")
                    break
            else:
                stuck_count = 0
                last_time = current_time_text
            
            try:
                current_time = parse_time(current_time_text)
                start_time = parse_time(target_start)
                end_time = parse_time(target_end)
                
                if start_time <= current_time <= end_time:
                    print(f"Reached target range: {current_time_text}")
                    return True
                    
                if current_time > start_time:
                    if "AM" in current_time_text and "11:59 AM" in current_time_text:
                        actions.move_to_element(slider_handle).click_and_hold()\
                              .move_by_offset(step_size * 2, 0).release().perform()
                    else:
                        actions.move_to_element(slider_handle).click_and_hold()\
                              .move_by_offset(step_size, 0).release().perform()
                else:
                    if "PM" in current_time_text and "12:00 PM" in current_time_text:
                        actions.move_to_element(slider_handle).click_and_hold()\
                              .move_by_offset(-step_size * 2, 0).release().perform()
                    else:
                        actions.move_to_element(slider_handle).click_and_hold()\
                              .move_by_offset(-step_size, 0).release().perform()
                
                time.sleep(2)
                current_attempt += 1
                
            except Exception as e:
                print(f"Error comparing times: {str(e)}")
                break
                
        print(f"Failed to reach range {target_start}-{target_end}, last seen time: {current_time_text}")
        return False
        
    except Exception as e:
        print(f"Error setting time range: {str(e)}")
        driver.save_screenshot(f"time_error_{target_start.replace(':', '_')}.png")
        return False

def parse_time(time_str):
    time_part, period = time_str.split()
    hours, minutes = map(int, time_part.split(':'))
    if period == 'PM' and hours != 12:
        hours += 12
    elif period == 'AM' and hours == 12:
        hours = 0
    return hours * 60 + minutes

def zoom_and_wait_for_shadows(lat, lng, driver, zoom_level=16, shadow_wait_time=15):
    try:
        create_output_directories()
        driver.get("https://shademap.app")
        close_popups(driver)
        
        search_box = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "input[placeholder*='Search']"))
        )
        search_box.send_keys(f"{lat}, {lng}", Keys.RETURN)
        time.sleep(5)

        print(f"Zooming to level {zoom_level}...")
        for _ in range(3):
            try:
                driver.execute_script("""
                    if (window.map) {
                        window.map.setZoom(arguments[0]);
                    } else if (window._map) {
                        window._map.setZoom(arguments[0]);
                    } else {
                        for (let key in window) {
                            if (key.toLowerCase().includes('map') && window[key].setZoom) {
                                window[key].setZoom(arguments[0]);
                                break;
                            }
                        }
                    }
                """, zoom_level)
                print(f"Zoomed to level {zoom_level}")
                break
            except:
                time.sleep(1)        

        print(f"Waiting {shadow_wait_time} seconds for shadows to render...")
        time.sleep(shadow_wait_time)

        time_ranges = [
            ('05:00 AM', '06:00 AM'),
            ('06:00 AM', '07:00 AM'),
            ('07:00 AM', '08:00 AM'),
            ('08:00 AM', '09:00 AM'),
            ('09:00 AM', '10:00 AM'),
            ('10:00 AM', '11:00 AM'),
            ('11:00 AM', '12:00 PM'),
            ('12:00 PM', '01:00 PM'),
            ('01:00 PM', '02:00 PM'),
            ('02:00 PM', '03:00 PM'),
            ('03:00 PM', '04:00 PM'),
            ('04:00 PM', '05:00 PM'),
            ('05:00 PM', '06:00 PM'),
            ('06:00 PM', '07:00 PM')
        ]
        
        results = {}
        
        for start_time, end_time in time_ranges:
            if set_time_to_range(start_time, end_time,driver):
                time.sleep(5)
                time_label = f"{start_time.split()[0]}-{end_time.split()[0]} {end_time.split()[1]}"
                
                # Capture and save screenshot
                screenshot_path = capture_and_save_screenshot(time_label,driver)
                print(f"Screenshot saved to: {screenshot_path}")
                
                # Analyze and save analysis
                coverage, analysis_path = analyze_shadows(screenshot_path, time_label)
                
                if coverage is not None:
                    results[time_label] = {
                        'coverage': coverage,
                        'screenshot': screenshot_path,
                        'analysis': analysis_path
                    }
                    print(f"Shadow coverage for {time_label}: {coverage:.2f}%")
                    print(f"Analysis image saved to: {analysis_path}")
        
        return results
        
    except Exception as e:
        print(f"Error processing location: {str(e)}")
        driver.save_screenshot("error.png")
        raise

def capture_and_save_screenshot(time_label,driver):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        canvas = WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "canvas.maplibregl-canvas"))
        )
        
        # Create filename with timestamp and time label
        filename = f"screenshots/screenshot_{timestamp}_{time_label.replace(':', '_').replace(' ', '_')}.png"
        canvas.screenshot(filename)
        img = Image.open(filename)
        width, height = img.size
        
        # Calculate square crop dimensions
        left = (width - height) // 2
        right = left + height
        cropped_img = img.crop((left, 0, right, height))
        
        cropped_path = f"screenshots/croppedscreenshot_{timestamp}_{time_label.replace(':', '_').replace(' ', '_')}.png"
        cropped_img.save(cropped_path)
        print(f"Screenshot cropped and saved to {cropped_path}")
        
        return cropped_path
        
    except Exception as e:
        print(f"Screenshot failed: {str(e)}")
        error_filename = f"screenshots/error_screenshot_{time_label.replace(':', '_').replace(' ', '_')}.png"
        driver.save_screenshot(error_filename)
        return error_filename

def analyze_shadows(image_path, time_slot):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to load image")
        
        roof_color = [170, 170, 170]
        shadow_color = [137, 119, 110]
        tolerance = 10
        
        roof_mask = cv2.inRange(img, 
                              np.array(roof_color) - tolerance,
                              np.array(roof_color) + tolerance)
        
        shadow_mask = cv2.inRange(img,
                                np.array(shadow_color) - tolerance,
                                np.array(shadow_color) + tolerance)
        
        roof_only = np.count_nonzero(roof_mask)
        shaded = np.count_nonzero(shadow_mask)
        total_roof = roof_only + shaded
        
        print(f"Total roof pixels: {total_roof}")
        print(f"Shaded pixels: {shaded}")
        
        coverage = (shaded / total_roof) * 100 if total_roof > 0 else 0
        
        result = img.copy()
        result[shadow_mask > 0] = [0, 0, 255]
        result[roof_mask > 0] = [0, 255, 0]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_path = f"shadow_analysis/analysis_{timestamp}_{time_slot.replace(':', '_').replace(' ', '_')}.png"
        
        cv2.imwrite(analysis_path, result)
        
        return coverage, analysis_path
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return 0, None


def final_analyze_shadows(lat, lng):
    """Helper function to analyze shadows (not a Flask route)."""
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        shadow_results = zoom_and_wait_for_shadows(lat, lng, driver)
        shadow_percentages = [data['coverage'] for data in shadow_results.values()]
        return shadow_percentages
    except Exception as e:
        logger.error(f"Shadow analysis error: {str(e)}")
        return [0] * 24  # Fallback: assume no shadows if analysis fails
    finally:
        if 'driver' in locals():
            driver.quit()

            
    
# Temperature ratios data (from your CSV)
TEMPERATURE_RATIOS = {
    (0, 0): 0.9076088180828692,
    (0, 15): 0.9073964962763456,
    (0, 30): 0.9060206000493174,
    (0, 45): 0.8999732261460454,
    (1, 0): 0.9002457126981703,
    (1, 15): 0.8953573652884573,
    (1, 30): 0.907229252176094,
    (1, 45): 0.9311662607628155,
    (2, 0): 0.9233739906673412,
    (2, 15): 0.9332719570018436,
    (2, 30): 0.9453836468456469,
    (2, 45): 0.9618479979155142,
    (3, 0): 0.9747295315902657,
    (3, 15): 0.9746366990810451,
    (3, 30): 0.9648149477929403,
    (3, 45): 0.953000493509287,
    (4, 0): 0.9153543926607103,
    (4, 15): 0.8945345119547289,
    (4, 30): 0.9081522683480993,
    (4, 45): 0.9289179501351592,
    (5, 0): 0.932779492021194,
    (5, 15): 0.9454679269218842,
    (5, 30): 0.9639789295019678,
    (5, 45): 0.9509033375578244,
    (6, 0): 0.9218841488795524,
    (6, 15): 0.9309427874617974,
    (6, 30): 0.954812469864359,
    (6, 45): 1.0092263935586634,
    (7, 0): 1.1079272718116144,
    (7, 15): 1.164108841213367,
    (7, 30): 1.18475700335901,
    (7, 45): 1.1717520474056469,
    (8, 0): 1.2357650066289227,
    (8, 15): 1.368642797827531,
    (8, 30): 1.5254205682716955,
    (8, 45): 1.456551939418359,
    (9, 0): 1.6295312194287976,
    (9, 15): 1.6656103840956404,
    (9, 30): 1.4222739778376459,
    (9, 45): 1.4910898503701069,
    (10, 0): 1.835267654139172,
    (10, 15): 1.6557364694846037,
    (10, 30): 1.6779319482806831,
    (10, 45): 1.5265559368523431,
    (11, 0): 1.6213575303546452,
    (11, 15): 1.6549803246557098,
    (11, 30): 1.6456315844428517,
    (11, 45): 1.552572347746936,
    (12, 0): 1.5710906969999225,
    (12, 15): 1.5819983385605738,
    (12, 30): 1.6285233467855963,
    (12, 45): 1.5631065049970243,
    (13, 0): 1.520739289350351,
    (13, 15): 1.4466279371650406,
    (13, 30): 1.4349087222067807,
    (13, 45): 1.4540730601260303,
    (14, 0): 1.475094951595809,
    (14, 15): 1.612346076794919,
    (14, 30): 1.5996749517787585,
    (14, 45): 1.4131821938711504,
    (15, 0): 1.35816756579928,
    (15, 15): 1.3008584465132074,
    (15, 30): 1.3587327221327203,
    (15, 45): 1.2756146767518068,
    (16, 0): 1.2603665937064676,
    (16, 15): 1.2016387268443354,
    (16, 30): 1.2015327107874976,
    (16, 45): 1.1641233718714123,
    (17, 0): 1.101167908024493,
    (17, 15): 1.0766344289871525,
    (17, 30): 1.0395456941001504,
    (17, 45): 0.9763761189548524,
    (18, 0): 0.9708637225091843,
    (18, 15): 0.9808295521426784,
    (18, 30): 0.9575538253653317,
    (18, 45): 0.9310675344722648,
    (19, 0): 0.9245824523381043,
    (19, 15): 0.930004664185783,
    (19, 30): 0.9528684449442707,
    (19, 45): 0.9463648576908308,
    (20, 0): 0.895316810528872,
    (20, 15): 0.8857214544950232,
    (20, 30): 0.9136553198553261,
    (20, 45): 0.921738949215201,
    (21, 0): 0.9315583178765541,
    (21, 15): 0.9327614283096541,
    (21, 30): 0.934182341464704,
    (21, 45): 0.940530557134876,
    (22, 0): 0.9403220705296673,
    (22, 15): 0.936019383692655,
    (22, 30): 0.9423262473475191,
    (22, 45): 0.9553520107991081,
    (23, 0): 0.9552661921992273
}



def get_monthly_demand_data(month_name, area):
    try:
        with open('final_data.csv', 'r') as f:
            csv_data = f.read()
        
        reader = csv.DictReader(StringIO(csv_data))
        for row in reader:
            if row['Month'].lower() == month_name.lower():
                hourly_data = []
                for hour in range(1, 25):
                    key = str(hour)
                    power_kw = float(row[key])  # Multiply by area
                    hourly_data.append({
                        'hour': hour - 1,  # 0-23 format
                        'power_kw': power_kw,
                        'power_kw_per_m2': float(row[key])  # Keep original kW/m² value
                    })
                return hourly_data
        return None
    except Exception as e:
        logger.error(f"Error reading demand data: {str(e)}")
        return None



def get_buildings_in_radius(lat, lng, radius):
    """Query buildings in radius with validation"""
    # Validate coordinates and radius
    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        return {'error': 'Invalid coordinates', 'count': 0, 'total_roof_area': 0.0, 'buildings': []}
    if radius <= 0 or radius > 5000:
        return {'error': 'Radius must be between 1 and 5000 meters', 'count': 0, 'total_roof_area': 0.0, 'buildings': []}

    query = f"""
    [out:json];
    (
      way["building"](around:{radius},{lat},{lng});
    );
    out body;
    >;
    out skel qt;
    """
    try:
        result = overpass_api.query(query)
        buildings = []
        total_area = 0.0
        for way in result.ways:
            if len(way.nodes) >= 3:
                area = calculate_polygon_area(way.nodes)
                buildings.append({
                    'id': way.id,
                    'nodes': len(way.nodes),
                    'area': area,
                    'coordinates': [(float(node.lat), float(node.lon)) for node in way.nodes]
                })
                total_area += area
        return {'count': len(buildings), 'total_roof_area': total_area, 'buildings': buildings}
    except Exception as e:
        return {'error': str(e), 'count': 0, 'total_roof_area': 0.0, 'buildings': []}

def calculate_polygon_area(nodes):
    """Calculate area using shoelace formula with validation"""
    if len(nodes) < 3: return 0.0
    earth_radius = 6378137  # meters
    coords = []
    for node in nodes:
        try:
            lat = float(node.lat)
            lon = float(node.lon)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                coords.append((math.radians(lat), math.radians(lon)))
        except (ValueError, AttributeError):
            continue
    
    if len(coords) < 3: return 0.0
    
    area = 0.0
    for i in range(len(coords)):
        j = (i + 1) % len(coords)
        area += (coords[j][1] - coords[i][1]) * (2 + math.sin(coords[i][0]) + math.sin(coords[j][0]))
    return abs(area) * (earth_radius ** 2) / 2

@app.route('/predict_power', methods=['POST'])
def predict_power():
    if not model:
        return jsonify({'error': "Model not loaded"}), 500
        
    try:
        data = request.get_json()
        if not data or 'sequences' not in data or 'area' not in data:
            return jsonify({'error': 'Missing required data'}), 400
        
        lat = float(data.get('lat'))
        lng = float(data.get('lng'))        
        sequences = data.get('sequences', [])
        area = float(data.get('area', 0))
        shadow_percentages = [0,0,0,0,0,0] + final_analyze_shadows(lat, lng) + [0,0,0,0]

            
        if not isinstance(sequences, list) or len(sequences) == 0:
            return jsonify({'error': 'Invalid sequences format'}), 400

        hourly_predictions = []
        for i, sequence in enumerate(sequences):
            try:
                if not sequence or len(sequence) != window_size:
                    logger.warning(f"Invalid sequence length {len(sequence)} for hour {i}")
                    hourly_predictions.append(0)
                    continue

                # Convert sequence to numpy array
                np_sequence = np.array(sequence, dtype=np.float32)
                
                # Log the GHI values for this hour
                ghi_values = [f"{x[0]:.2f}" for x in sequence]  # Extract GHI values
                logger.info(f"Hour {i:02d} GHI inputs (kW/m²): {', '.join(ghi_values)}")
                
                np_sequence = np_sequence.reshape(1, window_size, 3)  # 3 features
                if np_sequence[0][0][0] <= 10**-6:
                    hourly_predictions.append(0)
                else:
                    prediction = model.predict(np_sequence, verbose = 0)
                    area_s=area*(shadow_percentages[i]/100)#percentage here
                    raw_kw = float(prediction[0][0])/136227
                    start_value = 0.20
                    end_value = 0.80
                    num_points = int(area_s//1.96)


                    eff = np.linspace(start_value, end_value, num_points)
                    print("raw_kw",raw_kw)
                    print("eff",eff)
                    raw_power_arr=np.array(raw_kw)
                    print("raw_power_arr",raw_power_arr)
                    power_shadow = np.sum(raw_power_arr*eff)/num_points
                    power_shadow=power_shadow

                    
                    print("power_shadow",power_shadow)
                    area_ns=area-area_s
                    print("area_ns",area_ns)
                    
                    adjusted_kw = (raw_kw * area_ns)
                    print("adjusted_kw",adjusted_kw)
                    total_power=(power_shadow+adjusted_kw)
                    print("total_power",total_power)



                      # Adjust for area
                    hourly_predictions.append(max(0, total_power))
                    
            except Exception as e:
                logger.error(f"Error processing hour {i}: {str(e)}")
                hourly_predictions.append(0)

        return jsonify({
            'hourly_predictions': hourly_predictions,
            'total_prediction': sum(hourly_predictions),
            'units': 'kW'
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_demand_curve', methods=['POST'])
def get_demand_curve():
    try:
        data = request.get_json()
        month = data.get('month')
        area = float(data.get('area', 1.0))  # Default to 1 m² if not provided
        
        if not month:
            return jsonify({'error': 'Month parameter is required'}), 400
        
        demand_data = get_monthly_demand_data(month, area)
        if not demand_data:
            return jsonify({'error': f'No data found for month: {month}'}), 404
            
        return jsonify({
            'month': month,
            'hourly_demand': demand_data,
            'units': 'kW'
        })
    except Exception as e:
        logger.error(f"Demand curve error: {str(e)}")
        return jsonify({'error': str(e)}), 500




    
@app.route('/get_15min_solar', methods=['POST'])
def get_15min_solar():
    try:
        data = request.get_json()
        lat = data.get('lat')
        lng = data.get('lng')
        date = data.get('date')
        
        # Validate coordinates
        try:
            lat = float(lat)
            lng = float(lng)
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                return jsonify({'error': 'Coordinates out of valid range'}), 400
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid coordinate format'}), 400

        params = {
            'latitude': lat,
            'longitude': lng,
            'hourly': 'shortwave_radiation,temperature_2m',
            'timezone': 'auto',
            'start_date': date,
            'end_date': date
        }
        
        response = requests.get('https://api.open-meteo.com/v1/forecast', params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly_data = []
        if 'hourly' in data:
            for i in range(len(data['hourly']['time'])):
                try:
                    time = datetime.fromisoformat(data['hourly']['time'][i])
                    hour = time.hour
                    minute = time.minute
                    ratio = TEMPERATURE_RATIOS.get((hour, minute), 1.0)
                    
                    hourly_data.append({
                        'time': time.isoformat(),
                        'irradiation': float(data['hourly']['shortwave_radiation'][i]),
                        'ambient_temp': float(data['hourly']['temperature_2m'][i]),
                        'module_temp': float(data['hourly']['temperature_2m'][i]) * ratio,
                        'hour': hour,
                        'minute': minute
                    })
                except (ValueError, KeyError) as e:
                    continue
        
        if not hourly_data:
            return jsonify({'error': 'No valid solar data found'}), 400
            
        return jsonify({
            'date': date,
            'hourly': hourly_data
        })

    except Exception as e:
        logger.error(f"15min solar data error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_solar', methods=['POST'])
def get_solar():
    try:
        data = request.get_json()
        lat = data.get('lat')
        lng = data.get('lng')
        custom_date = data.get('date')
        
        # Validate coordinates
        try:
            lat = float(lat)
            lng = float(lng)
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                return jsonify({'error': 'Coordinates out of valid range'}), 400
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid coordinate format'}), 400

        target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d') if not custom_date else custom_date
        
        params = {
            'latitude': lat,
            'longitude': lng,
            'hourly': 'direct_normal_irradiance,diffuse_radiation',
            'timezone': 'auto',
            'start_date': target_date,
            'end_date': target_date
        }
        
        response = requests.get('https://api.open-meteo.com/v1/forecast', params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly_data = []
        if 'hourly' in data:
            for i in range(len(data['hourly']['time'])):
                time = datetime.fromisoformat(data['hourly']['time'][i])
                dni = data['hourly']['direct_normal_irradiance'][i]
                dhi = data['hourly']['diffuse_radiation'][i]
                ghi = (dni * math.cos(math.radians(90-50))) + dhi
                hourly_data.append({
                    'time': time.isoformat(),
                    'dni': dni,
                    'dhi': dhi,
                    'ghi': ghi,
                    'hour': time.hour
                })
        
        return jsonify({
            'date': target_date,
            'hourly': hourly_data
        })

    except Exception as e:
        logger.error(f"Solar data error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_buildings', methods=['POST'])
def get_buildings():
    try:
        data = request.get_json()
        lat = data.get('lat')
        lng = data.get('lng')
        radius = data.get('radius')
        
        # Validate inputs
        try:
            lat = float(lat)
            lng = float(lng)
            radius = float(radius)
            if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                return jsonify({'error': 'Coordinates out of valid range'}), 400
            if radius <= 0 or radius > 5000:
                return jsonify({'error': 'Radius must be between 1 and 5000 meters'}), 400
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid input format'}), 400

        result = get_buildings_in_radius(lat, lng, radius)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Building data error: {str(e)}")
        return jsonify({
            'error': f"Error processing request: {str(e)}",
            'count': 0,
            'total_roof_area': 0.0,
            'buildings': []
        })




@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
