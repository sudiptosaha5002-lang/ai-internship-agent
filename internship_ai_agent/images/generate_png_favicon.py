import math
from PIL import Image, ImageDraw

def main():
    # Create a 128x128 image with an alpha channel
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Helper function to interpolate colors between Cyan #00f0ff (0, 240, 255) and Emerald #10b981 (16, 185, 129)
    def get_gradient_color(step, total_steps):
        r = int(0 + (16 - 0) * (step / total_steps))
        g = int(240 + (185 - 240) * (step / total_steps))
        b = int(255 + (129 - 255) * (step / total_steps))
        return (r, g, b, 255)

    # 1. Draw glowing background hexagon
    def get_hexagon_pts(cx, cy, r):
        pts = []
        for i in range(6):
            angle = math.radians(30 + i * 60)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            pts.append((x, y))
        return pts

    # Draw glowing outer hexagon (with thick stroke simulating glow)
    for i in range(5):
        r = 50 + i * 2
        opacity = int(255 * (1.0 - i * 0.2))
        pts = get_hexagon_pts(64, 64, r)
        draw.polygon(pts, outline=(0, 240, 255, opacity // 4), width=6 - i)

    # Core hexagon stroke
    pts = get_hexagon_pts(64, 64, 48)
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % 6]
        color = get_gradient_color(i, 6)
        draw.line([p1, p2], fill=color, width=4)

    # 2. Draw outer dashed orbit rings (dots)
    for angle_deg in range(0, 360, 45):
        angle = math.radians(angle_deg)
        x = 64 + 32 * math.cos(angle)
        y = 64 + 32 * math.sin(angle)
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(16, 185, 129, 200))

    # 3. Draw central node
    draw.ellipse([64 - 10, 64 - 10, 64 + 10, 64 + 10], fill=(0, 240, 255, 255))
    draw.line([64 - 20, 64, 64 + 20, 64], fill=(0, 240, 255, 255), width=3)

    # 4. Draw orbital signal spark nodes
    # Top right
    draw.ellipse([64 + 28 - 4, 64 - 16 - 4, 64 + 28 + 4, 64 - 16 + 4], fill=(0, 240, 255, 255))
    # Bottom left
    draw.ellipse([64 - 28 - 4, 64 + 16 - 4, 64 - 28 + 4, 64 + 16 + 4], fill=(16, 185, 129, 255))

    # Save image
    img.save("internship_ai_agent/images/favicon.png")
    print("favicon.png generated successfully!")

if __name__ == "__main__":
    main()
