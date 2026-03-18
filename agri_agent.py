import pygame as pg
import math, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ────────────────────────────────────────────────────────────────
#  CONSTANTS
# ────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1280, 740
PLAY_W        = 920 
TRACTOR_SIZE  = 28
BLOCK_SIZE    = 30
SPACE         = 1
FPS           = 60

WHEAT_GROW_1_TO_2 = 1 / 300
WHEAT_GROW_2_TO_3 = 1 / 500

N_AGENTS = 5

# Colors
C_BG          = (15,   20,  15)
C_PANEL       = (22,   30,  22)
C_STAGE1      = (55,  120,  40)
C_STAGE2      = (180, 165,  35)
C_STAGE3      = (205, 130,  20)
C_HARVESTED   = (30,   30,  30)
C_TRACTOR_AI  = (30,   90,  30)
C_TRACTOR_USR = (20,   60, 160)
C_TRACTOR_EDG = (10,   40,  10)
C_ARROW       = (220, 220, 220)
C_DELIVERY    = (200, 200,  20)
C_DELIVERY_BG = ( 60,  60,   0)
C_FUEL_BAR    = ( 50, 200,  80)
C_FUEL_LOW    = (220,  60,  40)
C_WHEAT_BAR   = (200, 170,  30)
C_MONEY_TEXT  = (180, 220, 100)
C_RAY_HIT     = (255, 100,  50, 120)
C_RAY_MISS    = ( 80, 180, 100,  40)
C_TEXT        = (200, 220, 190)
C_TEXT_DIM    = (100, 130, 100)
C_OVERLAY     = ( 30,  30,  30, 180)
C_AGENT_COLORS = [
    (30,   90,  30),
    (90,   30, 120),
    (30,   80, 160),
    (200,  60,  90),  
    (60,  180,  75),  
]

# FARM LAYOUT
BIG_BLOCKS = [
    pg.Rect(10, HEIGHT-10-216, 309, 216),
    pg.Rect(10+309+10, HEIGHT-10-216, 309, 216),
    pg.Rect(10, HEIGHT-10-216-10-216, 309, 210),
    pg.Rect(10+309+10, HEIGHT-10-216-10-216, 309, 216),
    pg.Rect(10, HEIGHT-10-216-10-216-10-247, 309, 247),
    pg.Rect(10+309+10, HEIGHT-10-216-10-216-10-247, 309, 247),
]

def _gen_small_blocks(big, bs=BLOCK_SIZE, sp=SPACE):
    out = []
    for i in range(big.height // (bs + sp)):
        for j in range(big.width // (bs + sp)):
            out.append(pg.Rect(
                big.x + j*(bs+sp),
                big.y + i*(bs+sp),
                bs, bs 
            ))

    return out

ALL_WHEAT_REACTS = []
for _b in BIG_BLOCKS:
    ALL_WHEAT_REACTS.extend(_gen_small_blocks(_b))

# ────────────────────────────────────────────────────────────────
# KEYBOARD
# ────────────────────────────────────────────────────────────────

def keys_to_actions(keys):
    if keys[pg.K_UP]:    return 0   
    if keys[pg.K_DOWN]:  return 1   
    if keys[pg.K_LEFT]:  return 2   
    if keys[pg.K_RIGHT]: return 3   
    if keys[pg.K_SPACE]: return 4   
    return 5 

# ────────────────────────────────────────────────────────────────
#  ENVIRONMENT
# ────────────────────────────────────────────────────────────────

class FarmAgent:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id
        self.reset()

    def reset(self):
        self.x     = float(random.randint(670, PLAY_W - TRACTOR_SIZE - 10))
        self.y     = float(random.randint(20, HEIGHT - TRACTOR_SIZE - 20))
        self.angle = float(random.randrange(0, 360, 45))

    @property
    def rect(self):
        return pg.Rect(int(self.x), int(self.y), TRACTOR_SIZE, TRACTOR_SIZE)

    def step(self, action):

        speed = 3

        if action == 0:
            rad = math.radians(self.angle)
            self.x += math.cos(rad) * speed
            self.y += math.sin(rad) * speed
        if action == 1:
            rad = math.radians(self.angle)
            self.x -= math.cos(rad) * speed
            self.y -= math.sin(rad) * speed
        if action == 2:
            self.angle = (self.angle - 45) % 360
        if action == 3:
            self.angle = (self.angle + 45) % 360

        self.x = max(0, min(PLAY_W - TRACTOR_SIZE, self.x))
        self.y = max(0, min(HEIGHT - TRACTOR_SIZE, self.y))

class FarmEnv:
    def __init__(self):
        self.regrow_wheat()

    def regrow_wheat(self):
        self.wheat_states = {
            i: 1 
            for i in range(len(ALL_WHEAT_REACTS))
        }

    def update_wheat(self):
        for idx in self.wheat_states:
            stage = self.wheat_states[idx]

            if stage == 1 and random.random() < WHEAT_GROW_1_TO_2:
                self.wheat_states[idx] = 2
            elif stage == 2 and random.random() < WHEAT_GROW_2_TO_3:
                self.wheat_states[idx] = 3
        

# ────────────────────────────────────────────────────────────────
#  RENDERER
# ────────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self):
        pg.init()

        self.win = pg.display.set_mode((WIDTH, HEIGHT))
        self.clk = pg.time.Clock()

        self.mono_11 = pg.font.SysFont("monospace", 11)

        self.action_cooldown = 0

    def get_human_action (self):
        self.action_cooldown -= 1

        keys = pg.key.get_pressed()
        act = keys_to_actions(keys)
        if act in (2, 3, 4) and self.action_cooldown > 0:
            return 5
        if act in (2, 3, 4):
            self.action_cooldown = 8
        return act

    def poll_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
        return True

    def _draw_tractor(self, agnet, color, idx):
        r = agnet.rect
        pg.draw.rect(self.win, color, r, border_radius=4)
        pg.draw.rect(self.win, C_TRACTOR_EDG, r, 1, border_radius=4)

        rad    = math.radians(agnet.angle)
        cx, cy = r.centerx, r.centery
        size1  = TRACTOR_SIZE / 4
        size2  = TRACTOR_SIZE / 3
        d120   = math.radians(120)
        tip    = (cx + size2 * math.cos(rad),       cy + size2 * math.sin(rad))
        left   = (cx + size1 * math.cos(rad + d120), cy + size1 * math.sin(rad + d120))
        right  = (cx + size1 * math.cos(rad - d120), cy + size1 * math.sin(rad - d120))
        pg.draw.polygon(self.win, C_ARROW, [tip, left, right])

        lbl = self.mono_11.render(str(idx), True, (255, 255, 255))
        self.win.blit(lbl, (r.x + 2, r.y + 2))
    
    def _draw_wheat_block(self, rect, stage):
        colors = {1: C_STAGE1, 2: C_STAGE2, 3: C_STAGE3}
        c = colors.get(stage, C_HARVESTED)
        pg.draw.rect(self.win, c, rect, border_radius=2)
        edge = tuple(max(0, v - 30) for v in c)
        pg.draw.rect(self.win, edge, rect, 1, border_radius=2)

    
    def draw(self, agents, env):
        win = self.win
        win.fill(C_BG)

        for idx, rect in enumerate(ALL_WHEAT_REACTS):
            stage = env.wheat_states[idx]
            self._draw_wheat_block(rect, satge)


        for i, ag in enumerate(agents):
            color = C_AGENT_COLORS[i % len(C_AGENT_COLORS)]
            self._draw_tractor(ag, color, i)

        pg.display.flip()
        self.clk.tick(FPS)

    def close(self):
        pg.quit()
# ────────────────────────────────────────────────────────────────
#  TRANNING LOOP
# ────────────────────────────────────────────────────────────────

def train():
    env         = FarmEnv()
    agents      = [FarmAgent(i+1) for i in range(N_AGENTS)]
    humen_agent = FarmAgent(0)
    agents.append(humen_agent)
    renderer = Renderer()

    running = True

    while running:
        running = renderer.poll_events()
        if not renderer:
            break

        env.update_wheat()

        act = renderer.get_human_action ()
        humen_agent.step(act)


        renderer.draw(agents, env)

    renderer.close()
    print("Goodbye.")

if __name__ == "__main__":
    train()
