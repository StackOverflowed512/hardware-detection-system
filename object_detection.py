import cv2
import numpy as np
import os
import math
import argparse
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

class HardwareDetector:
    def __init__(self, model_path=None, pixel_to_mm=None):
        self.colors = {
            'screw': (0, 255, 0),      # Green
            'bolt': (0, 165, 255),     # Orange
            'nut': (255, 0, 0),        # Blue
            'washer': (0, 0, 255),     # Red
            'lock_washer': (255, 0, 255), # Purple
            'unknown': (128, 128, 128) # Gray
        }
        
        self.thresholds = {
            'min_area': 100,
            'circularity': {'washer': 0.7, 'nut': 0.6, 'min': 0.4},
            'hole_ratio': {'nut': 0.15, 'washer': 0.15, 'lock_washer': 0.1},
            'aspect_ratio': {'screw': 2.0, 'bolt': 1.8},
            'solidity': {'nut': 0.8, 'washer': 0.7},
        }
        
        # For display purposes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 2
        self.text_color = (255, 255, 255)
        self.line_thickness = 2
        
        # Pixel to mm conversion (can be calibrated)
        self.pixel_to_mm = pixel_to_mm if pixel_to_mm else 0.1
        
        # Classification model
        self.model = None
        self.classes = ['screw', 'bolt', 'nut', 'washer', 'lock_washer', 'unknown']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print("Using rule-based classification")

    def load_model(self, model_path):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def extract_features(self, contour, image):
        """Extract features from contour for classification"""
        if contour is None or len(contour) < 5:
            return [0] * 10
            
        # Basic shape features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-5)
        
        # Bounding rectangle features
        rect = cv2.minAreaRect(contour)
        _, (width, height), angle = rect
        
        # Ensure width and height are valid
        if width <= 0 or height <= 0:
            width, height = 1, 1
            
        aspect_ratio = max(width, height) / (min(width, height) + 1e-5)
        
        # Convexity features
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) if len(hull) >= 3 else area
        solidity = area / (hull_area + 1e-5)
        
        # Create mask for the object
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Check for holes
        has_hole, hole_ratio, holes = self.detect_holes(contour, image, mask)
        
        # Check if hexagonal
        is_hexagonal = self.is_hexagonal_shape(contour, perimeter)
        
        # Additional features
        hull_perimeter = cv2.arcLength(hull, True) if len(hull) >= 3 else perimeter
        convexity = perimeter / (hull_perimeter + 1e-5)
        
        # Rectangle fit
        rect_area = width * height
        rectangularity = area / (rect_area + 1e-5) if rect_area > 0 else 0
        
        return [
            area, perimeter, circularity, aspect_ratio, solidity,
            has_hole, hole_ratio, is_hexagonal, convexity, rectangularity
        ]

    def detect_holes(self, contour, image, mask):
        """Detect if the object has holes and calculate hole ratio"""
        has_hole, hole_ratio = 0, 0
        holes = []
        
        # Convert image to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create a mask for the object
        object_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Find holes using contour hierarchy
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours or no valid hierarchy, try alternative approach
        if len(contours) <= 1 or hierarchy is None:
            # Try adaptive threshold to find holes
            if np.sum(object_gray) > 0:
                # Get the non-zero pixels
                non_zero_pixels = object_gray[mask > 0]
                if len(non_zero_pixels) > 0:
                    mean_val = np.mean(non_zero_pixels)
                    if mean_val > 0:
                        # Threshold to find holes
                        threshold_value = int(0.6 * mean_val)
                        _, hole_mask = cv2.threshold(object_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
                        
                        # Clean up with morphology
                        kernel = np.ones((3, 3), np.uint8)
                        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel)
                        hole_mask = cv2.bitwise_and(hole_mask, mask)
                        
                        # Find hole contours
                        hole_contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Filter valid holes
                        area = cv2.contourArea(contour)
                        min_hole_area = area * 0.01
                        valid_holes = [h for h in hole_contours if cv2.contourArea(h) > min_hole_area]
                        
                        if valid_holes:
                            has_hole = 1
                            total_hole_area = sum(cv2.contourArea(h) for h in valid_holes)
                            hole_ratio = total_hole_area / area
                            holes = valid_holes
        else:
            # Use hierarchy to find holes (child contours)
            area = cv2.contourArea(contour)
            for i, (_, _, _, parent) in enumerate(hierarchy[0]):
                if parent >= 0:  # This is a hole (child contour)
                    hole_area = cv2.contourArea(contours[i])
                    if hole_area > area * 0.01:  # Minimum hole size
                        has_hole = 1
                        holes.append(contours[i])
            
            if has_hole:
                total_hole_area = sum(cv2.contourArea(h) for h in holes)
                hole_ratio = total_hole_area / area
        
        return has_hole, hole_ratio, holes

    def is_hexagonal_shape(self, contour, perimeter):
        """Check if the contour resembles a hexagon"""
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        return 5 <= len(approx) <= 7  # Allow for some imprecision

    def classify_object(self, features, contour=None, image=None):
        """Classify hardware object using features"""
        # Use ML model if available
        if self.model is not None:
            try:
                features_array = np.array(features).reshape(1, -1)
                prediction = self.model.predict(features_array)[0]
                return prediction
            except Exception as e:
                print(f"ML classification failed: {e}. Using rule-based.")
        
        # Fall back to rule-based classification
        area, perimeter, circularity, aspect_ratio, solidity, has_hole, hole_ratio, is_hexagonal, _, rectangularity = features
        
        if area <= 0 or perimeter <= 0:
            return 'unknown'

        # Washer detection
        if has_hole and circularity > self.thresholds['circularity']['washer'] and 0.1 < hole_ratio < 0.7:
            return 'washer'
            
        # Nut detection
        if has_hole and is_hexagonal and solidity > self.thresholds['solidity']['nut']:
            return 'nut'
            
        # Lock washer detection
        if has_hole and 0.4 < circularity < 0.75 and hole_ratio > 0.1:
            return 'lock_washer'
            
        # Screw/bolt detection
        if aspect_ratio > self.thresholds['aspect_ratio']['screw']:
            if has_hole and hole_ratio > 0.1:  # Threaded hole
                return 'bolt'
            return 'screw'
        elif aspect_ratio > self.thresholds['aspect_ratio']['bolt']:
            return 'bolt'
            
        # Check for regular nuts without clear holes (due to lighting/angle)
        if is_hexagonal and solidity > 0.75:
            return 'nut'
            
        # Default detection
        if has_hole and circularity > 0.6:
            return 'washer'
            
        return 'unknown'

    def measure_dimensions(self, contour, object_type):
        """Measure dimensions of the hardware in pixels, then convert to mm"""
        if contour is None or len(contour) < 5:
            return None
            
        # Get bounding rectangle
        rect = cv2.minAreaRect(contour)
        center, (width, height), angle = rect
        
        # Convert to mm
        width_mm = width * self.pixel_to_mm
        height_mm = height * self.pixel_to_mm
        
        # Get object-specific measurements
        if object_type == 'washer':
            # For washer, find outer and inner diameter
            mask = np.zeros((500, 500), dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Use minimum enclosing circle for outer diameter
            (x, y), outer_radius = cv2.minEnclosingCircle(contour)
            outer_diameter = 2 * outer_radius * self.pixel_to_mm
            
            # Inner diameter (approximated)
            inner_diameter = outer_diameter * 0.5  # Estimate if actual hole not detected
            
            return {
                'outer_diameter': round(outer_diameter, 1),
                'inner_diameter': round(inner_diameter, 1)
            }
            
        elif object_type == 'nut':
            # For nuts, measure across flats and estimate thread size
            # Use minimum width as across flats
            across_flats = min(width, height) * self.pixel_to_mm
            
            # Estimate thread size (approximated from across flats)
            # Common conversion: thread ≈ across_flats * 0.8
            thread_size = across_flats * 0.8
            
            return {
                'across_flats': round(across_flats, 1),
                'thread_size': round(thread_size, 1)
            }
            
        elif object_type in ['screw', 'bolt']:
            # For screws/bolts, measure length and diameter
            length = max(width, height) * self.pixel_to_mm
            diameter = min(width, height) * self.pixel_to_mm
            
            return {
                'length': round(length, 1),
                'diameter': round(diameter, 1)
            }
            
        else:
            # Generic measurements
            return {
                'width': round(width_mm, 1),
                'height': round(height_mm, 1)
            }

    def preprocess_image(self, image):
        """Preprocess image for better hardware detection"""
        if image is None or image.size == 0:
            return None, None
            
        # Create a copy and convert to grayscale
        img_copy = image.copy()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_copy.copy()
            img_copy = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Enhance contrast
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
        except Exception:
            enhanced = gray
        
        # Remove noise
        try:
            blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
        except Exception:
            blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Multi-threshold approach
        try:
            # Global threshold
            _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Adaptive threshold
            thresh2 = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Combine thresholds
            thresh = cv2.bitwise_or(thresh1, thresh2)
        except Exception:
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up with morphology
        kernel = np.ones((3, 3), np.uint8)
        try:
            mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        except Exception:
            mask = thresh
        
        return mask, gray

    def detect_hardware(self, image, calibrate=False, reference_size=None):
        """Main method to detect and classify hardware in image"""
        if image is None:
            print("Invalid image")
            return None, []
        
        # Calibrate pixel to mm ratio if requested
        if calibrate and reference_size is not None:
            self.calibrate_pixel_mm(image, reference_size)
        
        # Preprocess the image
        mask, gray = self.preprocess_image(image)
        if mask is None:
            return image, []
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Process each hardware item
        results = []
        img_result = image.copy()
        
        for i, contour in enumerate(contours):
            # Skip small contours
            if cv2.contourArea(contour) < self.thresholds['min_area']:
                continue
            
            # Extract features
            features = self.extract_features(contour, image)
            
            # Classify object
            object_type = self.classify_object(features, contour, image)
            
            # Measure dimensions
            dimensions = self.measure_dimensions(contour, object_type)
            
            if dimensions:
                # Add to results
                results.append({
                    'id': i+1,
                    'type': object_type,
                    'dimensions': dimensions,
                    'contour': contour
                })
                
                # Draw on result image
                self.draw_result(img_result, contour, object_type, dimensions, i+1)
        
        return img_result, results

    def draw_result(self, image, contour, object_type, dimensions, item_id):
        """Draw detection results on the image"""
        # Get color for this object type
        color = self.colors.get(object_type, self.colors['unknown'])
        
        # Draw contour
        cv2.drawContours(image, [contour], 0, color, self.line_thickness)
        
        # Get bounding rectangle for text placement
        x, y, w, h = cv2.boundingRect(contour)
        
        # Format dimensions string
        dim_str = ""
        if object_type == 'washer':
            dim_str = f"OD: {dimensions['outer_diameter']}mm, ID: {dimensions['inner_diameter']}mm"
        elif object_type == 'nut':
            dim_str = f"AF: {dimensions['across_flats']}mm, Thread: {dimensions['thread_size']}mm"
        elif object_type in ['screw', 'bolt']:
            dim_str = f"L: {dimensions['length']}mm, D: {dimensions['diameter']}mm"
        else:
            dim_str = f"W: {dimensions['width']}mm, H: {dimensions['height']}mm"
        
        # Draw text
        cv2.putText(
            image, f"#{item_id}: {object_type.upper()}", 
            (x, y-10), self.font, self.font_scale, self.text_color, self.font_thickness
        )
        
        cv2.putText(
            image, dim_str, 
            (x, y+h+15), self.font, self.font_scale, self.text_color, self.font_thickness
        )
    
    def calibrate_pixel_mm(self, image, reference_size_mm):
        """Calibrate pixel to mm ratio using a reference object"""
        # Ask user to identify reference object
        print("Please identify the reference object in the image.")
        print("Press 'r' when ready, then select the reference object by clicking on it.")
        
        # This is a simplified version, in a real implementation:
        # - Show image
        # - Let user click on reference object
        # - Measure its size in pixels
        # - Calculate ratio using known size in mm
        
        # For now, use a default estimation
        self.pixel_to_mm = 0.1  # This is just a placeholder
        print(f"Calibration set: 1 pixel = {self.pixel_to_mm} mm")


# Main function for testing
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hardware Detector')
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate pixel to mm ratio')
    parser.add_argument('--reference', type=float, default=None, help='Reference size in mm')
    args = parser.parse_args()
    
    # Get image path
    image_path = args.image
    if image_path is None:
        # Try default path
        default_paths = ["hardware.jpg", "hardware_image.jpg", "hardware_parts.jpg"]
        for path in default_paths:
            if os.path.exists(path):
                image_path = path
                break
    
    # Check if image exists
    if image_path is None or not os.path.exists(image_path):
        print("Please provide a valid image path.")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return
    
    # Initialize detector
    detector = HardwareDetector(model_path=args.model)
    
    # Process image
    result_image, hardware_items = detector.detect_hardware(
        image, 
        calibrate=args.calibrate,
        reference_size=args.reference
    )
    
    # Print results
    print(f"Found {len(hardware_items)} hardware items:")
    for item in hardware_items:
        print(f"Item #{item['id']}: {item['type'].upper()}")
        print(f"  Dimensions: {item['dimensions']}")
        print()
    
    # Display result
    cv2.imshow("Hardware Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    output_path = "hardware_detected.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()