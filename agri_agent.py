import pygame as pg
import math, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# ────────────────────────────────────────────────────────────────
#  CONSTANTS
# ────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1180, 740
PLAY_WIDTH    = 920 
HUD_X         = 930
TRACTOR_SIZE  = 24
BLOCK_SIZE    = 30
SPACE         = 1
FPS           = 0

# Wheat
WHEAT_GROW_1_TO_2  = 1 / 1100
WHEAT_GROW_2_TO_3  = 1 / 1300
MAX_WHEAT_CAN_CARRY = 10
WHEAT_TOTAL_GOAL   = 100

# Fuel
MAX_FUEL           = 500
FUEL_MOVE          = 0.5      
FUEL_ROTATE        = 0.1        
FUEL_REFUEL_COST   = 30       
FUEL_REFUEL_AMOUNT = 300 

MAX_IDLE_STEPS = 5_000

N_AGENTS = 10

# Rewards
R_HARVEST_WHEAT_STAGE_3        = +1.0
R_HARVEST_WHEAT_STAGE_1_OR_2   = -1.0
R_HARVEST_WHEAT_STAGE_3_GREEDY = -1.0
R_CARRY_WHEAT_TO_DELIVERY      = +5.0
R_REFUEL_NO_MONEY              = -1.0

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
    (200, 180,  90),
    (90,   60, 200), 
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

DELIVERY_RECT = pg.Rect(700, HEIGHT // 2 - 55, 130, 110)

def _generate_small_blocks(big_block, block_size=BLOCK_SIZE, spacing=SPACE):
    out = []
    for i in range(big_block.height // (block_size + spacing)):
        for j in range(big_block.width // (block_size + spacing)):
            out.append(pg.Rect(
                big_block.x + j*(block_size+spacing),
                big_block.y + i*(block_size+spacing),
                block_size, block_size 
            ))
    return out

ALL_WHEAT_RECTS = []
for big_block in BIG_BLOCKS:
    ALL_WHEAT_RECTS.extend(_generate_small_blocks(big_block))

# Rays
N_RAYS     = 360
MAX_DIST   = 300.0
RAY_ANGLES = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False).astype(np.float32)

_WR  = np.array([[r.x, r.y, r.x + r.width, r.y + r.height]
                  for r in ALL_WHEAT_RECTS], dtype=np.float32)
_WX0, _WY0, _WX1, _WY1 = _WR[:, 0], _WR[:, 1], _WR[:, 2], _WR[:, 3]

OBS_SIZE  = 8 + N_RAYS * 2
N_ACTIONS = 6

def _ray_aabb_intersect(ox, oy, dx, dy, x0, y0, x1, y1):
    EPS = 1e-8
    safe_dx = np.where(np.abs(dx) > EPS, dx, EPS)
    safe_dy = np.where(np.abs(dy) > EPS, dy, EPS)
    
    inv_dx = 1.0 / safe_dx
    inv_dy = 1.0 / safe_dy
    
    tx0 = (x0 - ox) * inv_dx
    tx1 = (x1 - ox) * inv_dx
    ty0 = (y0 - oy) * inv_dy
    ty1 = (y1 - oy) * inv_dy
    
    t_enter = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
    t_exit  = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
    hit     = (t_exit >= 0) & (t_enter <= t_exit) & (t_enter >= 0)
    return np.where(hit, t_enter, np.inf)

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
# DDQN
# ────────────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    def __init__(self, observation_size, action_count, hidden_size=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),   nn.ReLU())
        self.value = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, action_count))

    def forward(self, x):
        h = self.shared(x); v = self.value(h); a = self.advantage(h)
        return v + a - a.mean(1, keepdim=True)

class ReplayBuffer:
    def __init__(self, capacity=80_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size, device):
        batch    = random.sample(self.buffer, batch_size)
        observations  = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32, device=device)
        actions  = torch.tensor([x[1] for x in batch], dtype=torch.long,    device=device)
        rewards  = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=device)
        next_observations = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32, device=device)
        dones = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=device)
        return observations, actions, rewards, next_observations, dones

    def __len__(self):
        return len(self.buffer)

class DDQNAgent:
    GAMMA=0.99; LR=1e-3; BATCH_SIZE=128; EPSILON_START=1.0; EPSILON_MIN=0.05
    EPSILON_DECAY=0.9997; TARGET_UPDATE=600; MIN_BUFFER_SIZE=1_500

    def __init__(self, device):
        self.device    = device
        self.online_network = DuelingDQN(OBS_SIZE, N_ACTIONS).to(device)
        self.target_network = DuelingDQN(OBS_SIZE, N_ACTIONS).to(device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer    = optim.Adam(self.online_network.parameters(), lr=self.LR)
        self.buffer    = ReplayBuffer()
        self.epsilon    = self.EPSILON_START
        self.step_count = 0

    def select_action(self, observation, use_greedy=False):
        if not use_greedy and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online_network(tensor).argmax(1).item())

    def learn(self):
        if len(self.buffer) < self.MIN_BUFFER_SIZE:
            return None
        observations, actions, rewards, next_observations, dones = self.buffer.sample(self.BATCH_SIZE, self.device)
        with torch.no_grad():
            next_actions = self.online_network(next_observations).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_observations).gather(1, next_actions).squeeze()
            target_q_values = rewards + self.GAMMA * next_q_values * (1 - dones)
        q_values    = self.online_network(observations).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), 10)
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)
        return loss.item()

    def save(self, file_path):
        torch.save({"online": self.online_network.state_dict(),
                    "target": self.target_network.state_dict(),
                    "epsilon": self.epsilon}, file_path)
        print(f"[SAVE] → {file_path}")

    def load(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.online_network.load_state_dict(checkpoint["online"])
        self.target_network.load_state_dict(checkpoint["target"])
        self.epsilon              = checkpoint.get("epsilon",  self.EPSILON_MIN)
        print(f"[LOAD] ← {file_path}  ε={self.epsilon:.3f}")

# ────────────────────────────────────────────────────────────────
#  ENVIRONMENT
# ────────────────────────────────────────────────────────────────

class FarmAgent:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id
        self.reset()

    def reset(self):
        self.x             = float(random.randint(670, PLAY_WIDTH - TRACTOR_SIZE - 10))
        self.y             = float(random.randint(20, HEIGHT - TRACTOR_SIZE - 20))
        self.angle         = float(random.randrange(0, 360, 45))
        self.fuel          = float(MAX_FUEL)
        self.wheat_carried = 0
        self.money         = 0
        self.idle_steps    = 0

    @property
    def rect(self):
        return pg.Rect(int(self.x), int(self.y), TRACTOR_SIZE, TRACTOR_SIZE)

    def step(self, action):
        done = False
        reward = 0.0
        self.idle_steps += 1

        speed = 3
        if self.fuel > 0:
            if action == 0:
                radians = math.radians(self.angle)
                self.x += math.cos(radians) * speed
                self.y += math.sin(radians) * speed
                self.fuel -= FUEL_MOVE
            if action == 1:
                radians = math.radians(self.angle)
                self.x -= math.cos(radians) * speed
                self.y -= math.sin(radians) * speed
                self.fuel -= FUEL_MOVE
            if action == 2:
                self.angle = (self.angle - 45) % 360
                self.fuel -= FUEL_ROTATE
            if action == 3:
                self.angle = (self.angle + 45) % 360
                self.fuel -= FUEL_ROTATE
        if action == 4:
            if self.money >= FUEL_REFUEL_COST:
                self.money -= FUEL_REFUEL_COST
                self.fuel = min(self.fuel + FUEL_REFUEL_AMOUNT, MAX_FUEL)
            else:
                reward += R_REFUEL_NO_MONEY

        self.x = max(0, min(PLAY_WIDTH - TRACTOR_SIZE, self.x))
        self.y = max(0, min(HEIGHT - TRACTOR_SIZE, self.y))

        state = self.get_state()

        if self.idle_steps > MAX_IDLE_STEPS:
            done = True
        return state, reward, done

    def get_state(self):
        normalized_x         = self.x / PLAY_WIDTH
        normalized_y         = self.y / HEIGHT
        normalized_fuel      = self.fuel / MAX_FUEL
        normalized_wheat     = self.wheat_carried / MAX_WHEAT_CAN_CARRY
        delta_x              = (self.x - DELIVERY_RECT.centerx) / PLAY_WIDTH
        delta_y              = (self.y - DELIVERY_RECT.centery) / HEIGHT
        delivery_vector      = np.array([delta_x, delta_y])
        angle_radians        = math.radians(self.angle)
        forward_vector       = np.array([math.cos(angle_radians), math.sin(angle_radians)])
        side_vector          = np.array([-math.sin(angle_radians), math.cos(angle_radians)])
        dot_forward          = np.dot(delivery_vector, forward_vector)
        dot_side             = np.dot(delivery_vector, side_vector)
        return np.array([normalized_x, normalized_y, 
                         normalized_fuel, normalized_wheat, 
                         delta_x, delta_y, dot_forward, dot_side])

class FarmEnvironment:
    def __init__(self):
        self.regrow_wheat()

    def regrow_wheat(self):
        self.wheat_states = {
            i: 1 
            for i in range(len(ALL_WHEAT_RECTS))
        }

    def update_wheat(self):
        for idx in self.wheat_states:
            stage = self.wheat_states[idx]
            if stage == 1 and random.random() < WHEAT_GROW_1_TO_2:
                self.wheat_states[idx] = 2
            elif stage == 2 and random.random() < WHEAT_GROW_2_TO_3:
                self.wheat_states[idx] = 3

    def handle_harvest(self, agent):
        reward = 0.0
        agent_rect = agent.rect
        for idx, wheat_rect in enumerate(ALL_WHEAT_RECTS):
            stage = self.wheat_states[idx]
            if stage > 0 and agent_rect.colliderect(wheat_rect):
                if stage == 3:
                    if agent.wheat_carried == MAX_WHEAT_CAN_CARRY:
                        reward += R_HARVEST_WHEAT_STAGE_3_GREEDY
                    else:
                        agent.wheat_carried += 1
                        reward += R_HARVEST_WHEAT_STAGE_3
                else:
                    reward += R_HARVEST_WHEAT_STAGE_1_OR_2
                self.wheat_states[idx] = 0
        return reward

    def handle_delivery(self, agent):
        reward = 0.0
        agent_rect = agent.rect
        if agent_rect.colliderect(DELIVERY_RECT.inflate(-2*TRACTOR_SIZE, -2*TRACTOR_SIZE)) and agent.wheat_carried > 0:
            earned = agent.wheat_carried * 10
            agent.money += earned
            reward += R_CARRY_WHEAT_TO_DELIVERY * agent.wheat_carried
            agent.wheat_carried = 0
        return reward

    def step_environment(self, agent):
        reward_harvest = self.handle_harvest(agent)
        reward_delivery = self.handle_delivery(agent)
        return reward_harvest + reward_delivery
        

# ────────────────────────────────────────────────────────────────
#  RENDERER
# ────────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self):
        pg.init()
        self.window = pg.display.set_mode((WIDTH, HEIGHT))
        self.clock = pg.time.Clock()
        self.font_small = pg.font.SysFont("monospace", 11)
        self.font_bold = pg.font.SysFont("monospace", 16, bold=True)
        self._ray_surf  = pg.Surface((PLAY_WIDTH, HEIGHT), pg.SRCALPHA)
        self.action_cooldown = 0

    def get_human_action(self):
        self.action_cooldown -= 1
        keys = pg.key.get_pressed()
        action = keys_to_actions(keys)
        if action in (2, 3, 4) and self.action_cooldown > 0:
            return 5
        if action in (2, 3, 4):
            self.action_cooldown = 8
        return action

    def poll_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
        return True

    def _draw_bar(self, x_pos, y_pos, width, height, fraction, full_color, low_color, bg_color=(40,40,40), label=""):
        pg.draw.rect(self.window, bg_color, (x_pos, y_pos, width, height), border_radius=3)
        fill_width = int(width * max(0.0, min(1.0, fraction)))
        color  = low_color if fraction < 0.25 else full_color
        if fill_width > 0:
            pg.draw.rect(self.window, color, (x_pos, y_pos, fill_width, height), border_radius=3)
        pg.draw.rect(self.window, (80, 80, 80), (x_pos, y_pos, width, height), 1, border_radius=3)
        if label:
            lbl = self.font_small.render(label, True, C_TEXT_DIM)
            self.window.blit(lbl, (x_pos + width + 5, y_pos))

    def _draw_tractor(self, agent, color, index):
        rect = agent.rect
        pg.draw.rect(self.window, color, rect, border_radius=4)
        pg.draw.rect(self.window, C_TRACTOR_EDG, rect, 1, border_radius=4)
        radians    = math.radians(agent.angle)
        center_x, center_y = rect.centerx, rect.centery
        size1  = TRACTOR_SIZE / 4
        size2  = TRACTOR_SIZE / 3
        angle_120   = math.radians(120)
        tip    = (center_x + size2 * math.cos(radians),       center_y + size2 * math.sin(radians))
        left   = (center_x + size1 * math.cos(radians + angle_120), center_y + size1 * math.sin(radians + angle_120))
        right  = (center_x + size1 * math.cos(radians - angle_120), center_y + size1 * math.sin(radians - angle_120))
        pg.draw.polygon(self.window, C_ARROW, [tip, left, right])
        label = self.font_small.render(str(index), True, (255, 255, 255))
        self.window.blit(label, (rect.x + 2, rect.y + 2))
    
    def _draw_wheat_block(self, rect, stage):
        colors = {1: C_STAGE1, 2: C_STAGE2, 3: C_STAGE3}
        color = colors.get(stage, C_HARVESTED)
        pg.draw.rect(self.window, color, rect, border_radius=2)
        edge = tuple(max(0, value - 30) for value in color)
        pg.draw.rect(self.window, edge, rect, 1, border_radius=2)

    def _draw_rays(self, agents, wheat_states):
        self._ray_surf.fill((0, 0, 0, 0))
        active_indices = [i for i, s in wheat_states.items() if s > 0]
        if active_indices:
            ax0 = _WX0[active_indices]
            ay0 = _WY0[active_indices]
            ax1 = _WX1[active_indices]
            ay1 = _WY1[active_indices]

        for agent in agents:
            cx = agent.x + TRACTOR_SIZE / 2
            cy = agent.y + TRACTOR_SIZE / 2

            all_angles = math.radians(agent.angle) + RAY_ANGLES 
            dx_arr = np.cos(all_angles)[:, None]                  
            dy_arr = np.sin(all_angles)[:, None]                 

            if active_indices:
                t_vals = _ray_aabb_intersect(cx, cy, dx_arr, dy_arr, ax0, ay0, ax1, ay1)  
                closest_wheat_idx = np.argmin(t_vals, axis=1)               
                closest_t = t_vals[np.arange(N_RAYS), closest_wheat_idx]    
            else:
                closest_t = np.full(N_RAYS, np.inf)
                closest_wheat_idx = np.zeros(N_RAYS, dtype=int)

            dx_arr = dx_arr[:, 0]  
            dy_arr = dy_arr[:, 0]

            for r in range(N_RAYS):
                t = closest_t[r]
                hit = t <= MAX_DIST
                if hit:
                    hit_stage = wheat_states[active_indices[closest_wheat_idx[r]]]
                    col = C_RAY_HIT if hit_stage == 3 else (200, 200, 80, 80)
                    ex, ey = cx + dx_arr[r] * t, cy + dy_arr[r] * t
                else:
                    col = C_RAY_MISS
                    ex, ey = cx + dx_arr[r] * MAX_DIST, cy + dy_arr[r] * MAX_DIST
                pg.draw.line(self._ray_surf, col, (int(cx), int(cy)), (int(ex), int(ey)), 1)

        self.window.blit(self._ray_surf, (0, 0))
            

    def draw(self, agents, environment, steps):
        window = self.window
        window.fill(C_BG)

        for index, wheat_rect in enumerate(ALL_WHEAT_RECTS):
            stage = environment.wheat_states[index]
            self._draw_wheat_block(wheat_rect, stage)

        pg.draw.rect(window, C_DELIVERY_BG, DELIVERY_RECT, border_radius=6)
        pg.draw.rect(window, C_DELIVERY, DELIVERY_RECT, 2, border_radius=6)

        self._draw_rays(agents, environment.wheat_states)

        delivery_label1 = self.font_small.render("DELIVER", True, C_DELIVERY)
        window.blit(delivery_label1, (DELIVERY_RECT.centerx - 28, DELIVERY_RECT.centery - 10))
        delivery_label2 = self.font_small.render("here →$", True, C_DELIVERY)
        window.blit(delivery_label2, (DELIVERY_RECT.centerx - 26, DELIVERY_RECT.centery + 4))

        panel_x = HUD_X
        pg.draw.rect(window, C_PANEL, (panel_x - 10, 0, WIDTH - panel_x + 10, HEIGHT))
        pg.draw.line(window, (40, 60, 40), (panel_x - 10, 0), (panel_x - 10, HEIGHT), 1)

        y_pos = 15
        separator = self.font_small.render("─" * 30, True, (50, 70, 50))
        window.blit(separator, (panel_x, y_pos)); y_pos += 18
        episode = 0
        total_wheat = 0
        window.blit(self.font_bold.render(f"Episode:   {episode}", True, C_TEXT), (panel_x, y_pos)); y_pos += 20
        window.blit(self.font_bold.render(f"Total WT:  {total_wheat} / {WHEAT_TOTAL_GOAL}", True, C_TEXT), (panel_x, y_pos)); y_pos += 20
        window.blit(self.font_bold.render(f"Idle:  {steps} / {MAX_IDLE_STEPS}", True, C_TEXT), (panel_x, y_pos)); y_pos += 20
        window.blit(separator, (panel_x, y_pos)); y_pos += 18

        for agent_index, agent in enumerate(agents):
            agent_tag = str(agent.agent_id)
            agent_color = C_AGENT_COLORS[agent_index % len(C_AGENT_COLORS)]
            window.blit(self.font_bold.render(f"Agent {agent_tag}", True, agent_color), (panel_x, y_pos)); y_pos += 18
            self._draw_bar(panel_x, y_pos, 120, 10, agent.fuel / MAX_FUEL, C_FUEL_BAR, C_FUEL_LOW, label=f"Fuel {int(agent.fuel)}")
            y_pos += 16
            window.blit(self.font_small.render(f"  Money: ${agent.money}", True, C_TEXT_DIM), (panel_x, y_pos)); y_pos += 16
            window.blit(self.font_small.render(f"  Wheat: {agent.wheat_carried}/{MAX_WHEAT_CAN_CARRY}", True, C_TEXT_DIM), (panel_x, y_pos)); y_pos += 16
            window.blit(separator, (panel_x, y_pos)); y_pos += 16
            self._draw_tractor(agent, agent_color, agent_index)

        pg.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pg.quit()

# ────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ────────────────────────────────────────────────────────────────

def train():
    environment = FarmEnvironment()
    agents      = []
    ai_agents   = [FarmAgent(i+1) for i in range(N_AGENTS)]
    human_agent = FarmAgent('YOU')
    agents.append(human_agent)
    agents.extend(ai_agents)
    renderer = Renderer()
    running = True

    while running:
        running = renderer.poll_events()
        if not running:
            break
        environment.update_wheat()
        action = renderer.get_human_action()
        state_h, reward_h, done_h = human_agent.step(action)
        reward = environment.step_environment(human_agent)
        for agent in ai_agents:
            action = random.randint(0, 4)
            state_ai, reward_ai, done_ai = agent.step(action)
            reward = environment.step_environment(agent)
        renderer.draw(agents, environment, human_agent.idle_steps)
        if done_h:
            environment.regrow_wheat()
            human_agent.reset()
            for agent in ai_agents:
                agent.reset()
        pg.display.set_caption(f"FPS: {renderer.clock.get_fps():.2f}")
    renderer.close()
    print("Goodbye.")

if __name__ == "__main__":
    train()
