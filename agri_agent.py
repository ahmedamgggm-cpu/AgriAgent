import pygame as pg
import sys
import random
import math

# ────────────────────────────────────────────────────────────────
# Autonomous agent:
# - Detects wheat
# - Moves to harvest it
# - Delivers it to the delivery station
# - Refuels when necessary
# ────────────────────────────────────────────────────────────────

# ──────── Constants ────────
WIDTH, HEIGHT = 1000, 700
FPS           = 60
TRACTOR_SIZE  = 30
BLOCK_SIZE    = 30
SPACE         = 1

# ──────── Colors ────────
COLOR_BG           = (20, 20,  20)   
COLOR_STAGE_1      = (72, 130, 55)
COLOR_STAGE_2      = (189, 174, 50)  
COLOR_STAGE_3      = (210, 140, 30)
COLOR_TRACTOR      = ( 34,  85,  34)   
COLOR_TRACTOR_EDGE = ( 18,  50,  18)  
COLOR_ARROW        = (200, 200,  200)   

# ──────── Farm Blocks ────────
BIG_BLOCKS = [
    pg.Rect(10, HEIGHT-10-210, 320, 210),
    pg.Rect(340, HEIGHT-10-210, 320, 210),
    pg.Rect(10, HEIGHT-10-430, 320, 210),
    pg.Rect(340, HEIGHT-10-430, 320, 210),
    pg.Rect(10, HEIGHT-10-670, 320, 240),
    pg.Rect(340, HEIGHT-10-670, 320, 240),
]

def generate_small_blocks(big_rect, block_size=BLOCK_SIZE, space=SPACE):
    """Divide a big block into smaller blocks with spacing."""
    small_blocks = []
    rows = big_rect.height // (block_size + space)
    cols = big_rect.width  // (block_size + space)
    for i in range(rows):
        for j in range(cols):
            x = big_rect.x + j * (block_size + space)
            y = big_rect.y + i * (block_size + space)
            small_blocks.append(pg.Rect(x, y, block_size, block_size))
    return small_blocks

SMALL_BLOCKS = []
for block in BIG_BLOCKS:
    SMALL_BLOCKS.extend(generate_small_blocks(block))

# ────────────────────────────────────────────────────────────────
# Helpers 
# ────────────────────────────────────────────────────────────────

def random_position(size):
    """Return a random position for the tractor."""
    x = random.randint(670, WIDTH - size - 10)
    y = random.randint(10, HEIGHT - size - 10)
    return x, y

# ────────────────────────────────────────────────────────────────
# Environment  
# ────────────────────────────────────────────────────────────────

class FarmEnv:
    def reset(self):
        tx, ty = random_position(TRACTOR_SIZE)
        self.tractor = pg.Rect(tx, ty, TRACTOR_SIZE, TRACTOR_SIZE)
        self.angle = random.randrange(0, 360, 45)

# ────────────────────────────────────────────────────────────────
# Renderer 
# ────────────────────────────────────────────────────────────────

def draw_rect(surface, rect, fill_color, edge_color, border_radius=3):
    pg.draw.rect(surface, fill_color, rect, border_radius=border_radius)
    pg.draw.rect(surface, edge_color, rect, 1, border_radius=border_radius)

def draw_arrow(surface, cx, cy, angle_deg, size, color):
    rad = math.radians(angle_deg)
    r = size * 0.4
    d90 = math.radians(90)
    d120 = math.radians(120)
    tip   = (cx + r * math.cos(rad + d90), cy + r * math.sin(rad + d90))
    left  = (cx + r * math.cos(rad + d90 + d120), cy + r * math.sin(rad + d90 + d120))
    right = (cx + r * math.cos(rad + d90 + 2*d120), cy + r * math.sin(rad + d90 + 2*d120))
    pg.draw.polygon(surface, color, [tip, left, right])

def draw_tractor(surface, rect, fill_color, edge_color, arrow_color, angle_deg, size):
    draw_rect(surface, rect, fill_color, edge_color)
    draw_arrow(surface, rect.centerx, rect.centery, angle_deg, size, arrow_color)

class Renderer:
    def __init__(self):
        self.win = pg.display.set_mode((WIDTH, HEIGHT))
        self.clock = pg.time.Clock()

    def poll_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
        return True

    def draw(self, env):
        self.win.fill(COLOR_BG)

        for rect in SMALL_BLOCKS:
            pg.draw.rect(self.win, COLOR_STAGE_1, rect)

        draw_tractor(self.win, env.tractor, COLOR_TRACTOR, COLOR_TRACTOR_EDGE, COLOR_ARROW, env.angle, TRACTOR_SIZE)

        pg.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pg.quit()

# ────────────────────────────────────────────────────────────────
# Traninng Loop 
# ────────────────────────────────────────────────────────────────

def main():
    pg.init()
    env = FarmEnv()
    renderer = Renderer()
    env.reset()

    running = True
    while running:
        running = renderer.poll_events()
        renderer.draw(env)

    renderer.close()

if __name__ == '__main__':
    main()
