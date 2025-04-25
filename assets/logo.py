from PIL import Image, ImageDraw, ImageFont
import os

def create_logo():
    # Create a new image with a transparent background
    img = Image.new('RGBA', (200, 200), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a circle (Earth)
    draw.ellipse((40, 40, 160, 160), fill=(30, 80, 180, 255), outline=(255, 255, 255, 255), width=3)
    
    # Draw zigzag lines (seismic waves)
    for i in range(30, 180, 30):
        points = []
        for x in range(10, 190, 10):
            y = i + (-8 if x % 20 == 0 else 8)
            points.append((x, y))
        draw.line(points, fill=(255, 60, 60, 255), width=3)
    
    # Save the image
    img.save('assets/earthquake_logo.png')
    
    return 'assets/earthquake_logo.png'

if __name__ == "__main__":
    # Check if the assets directory exists, create it if it doesn't
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    # Create the logo
    logo_path = create_logo()
    print(f"Logo created at {logo_path}") 