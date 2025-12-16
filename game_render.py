import gymnasium as gym
import pygame
import torch
from pathlib import Path

from agents import PPOAgent, Trainer
from agents import KeyboardAgent
from agents import PlayerQ
from agents import ReinforceAgentV2, PlayerV2
from agents.ddqn_agent import QNetwork
from pvz import config

class PVZ():
    def __init__(self, render=True, max_frames=1000):
        self.env = gym.make('gym_pvz:pvz-env-v2')
        self.max_frames = max_frames
        self.render = render

    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        return config.N_LANES * (config.LANE_LENGTH + 2)

    def num_actions(self):
        return self.env.action_space.n

    def play(self, agent):
        """ Play one episode and collect observations and rewards """
        # Gymnasium reset() returns (observation, info)
        observation, _info = self.env.reset()
        t = 0

        for t in range(self.max_frames):
            if (self.render):
                self.env.render()

            action = agent.decide_action(observation)
            # Gymnasium step() returns (observation, reward, terminated, truncated, info)
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            if done:
                break

    def get_render_info(self):
        return self.env._scene._render_info


def render(render_info):
    pygame.init()
    pygame.font.init()  # you have to call this at the start,
    # if you want to use this module.
    myfont = pygame.font.SysFont('calibri', 30)

    screen = pygame.display.set_mode((1450, 650))
    zombie_sprite = {"zombie": pygame.image.load("assets/zombie_scaled.png").convert_alpha(),
                     "zombie_cone": pygame.image.load("assets/zombie_cone_scaled.png").convert_alpha(),
                     "zombie_bucket": pygame.image.load("assets/zombie_bucket_scaled.png").convert_alpha(),
                     "zombie_flag": pygame.image.load("assets/zombie_flag_scaled.png").convert_alpha(), }
    plant_sprite = {"peashooter": pygame.image.load("assets/peashooter_scaled.png").convert_alpha(),
                    "sunflower": pygame.image.load("assets/sunflower_scaled.png").convert_alpha(),
                    "wallnut": pygame.image.load("assets/wallnut_scaled.png").convert_alpha(),
                    "potatomine": pygame.image.load("assets/potatomine_scaled.png").convert_alpha(),
                    "potatomine_init": pygame.image.load("assets/PotatomineInit.png").convert_alpha()}
    projectile_sprite = {"pea": pygame.image.load("assets/pea.png").convert_alpha()}
    clock = pygame.time.Clock()
    cell_size = 75
    offset_border = 100
    offset_y = int(0.8 * cell_size)
    cumulated_score = 0

    while render_info:
        clock.tick(config.FPS)
        screen.fill((130, 200, 100))
        frame_info = render_info.pop(0)

        # The grid
        for i in range(config.LANE_LENGTH + 1):
            pygame.draw.line(screen, (0, 0, 0), (offset_border + i * cell_size, offset_border),
                             (offset_border + i * cell_size, offset_border + cell_size * (config.N_LANES)), 1)
        for j in range(config.N_LANES + 1):
            pygame.draw.line(screen, (0, 0, 0), (offset_border, offset_border + j * cell_size),
                             (offset_border + cell_size * (config.LANE_LENGTH), offset_border + j * cell_size), 1)

        # The objects
        for lane in range(config.N_LANES):
            for zombie_name, pos, offset in frame_info["zombies"][lane]:
                zombie_name = zombie_name.lower()
                screen.blit(zombie_sprite[zombie_name], (offset_border + cell_size * (pos + offset) - zombie_sprite[zombie_name].get_width(),
                    offset_border + lane * cell_size + offset_y - zombie_sprite[zombie_name].get_height()))
            for plant_data in frame_info["plants"][lane]:
                # 解包植物数据
                if len(plant_data) == 3:
                    plant_name, pos, is_active = plant_data
                else:
                    plant_name, pos = plant_data
                    is_active = None
                
                plant_name = plant_name.lower()
                # 土豆地雷根据激活状态选择图片
                if plant_name == "potatomine" and is_active is False:
                    sprite_key = "potatomine_init"
                else:
                    sprite_key = plant_name
                
                screen.blit(plant_sprite[sprite_key], (offset_border + cell_size * pos, 
                    offset_border + lane * cell_size + offset_y - plant_sprite[sprite_key].get_height()))
            for projectile_name, pos, offset in frame_info["projectiles"][lane]:
                projectile_name = projectile_name.lower()
                screen.blit(projectile_sprite[projectile_name],
                            (offset_border + cell_size * (pos + offset) - projectile_sprite[
                                projectile_name].get_width(),
                             offset_border + lane * cell_size))

        # Text
        sun_text = myfont.render('Sun: ' + str(frame_info["sun"]), False, (0, 0, 0))
        screen.blit(sun_text, (50, 600))
        cumulated_score += frame_info["score"]
        score_text = myfont.render('Score: ' + str(cumulated_score), False, (0, 0, 0))
        screen.blit(score_text, (200, 600))
        cooldowns_text = myfont.render('Cooldowns: ' + str(frame_info["cooldowns"]), False, (0, 0, 0))
        screen.blit(cooldowns_text, (350, 600))
        time = myfont.render('Time: ' + str(frame_info["time"]), False, (0, 0, 0))
        screen.blit(time, (900, 100))

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                render_info = []

        pygame.display.flip()

    pygame.quit()


agent_type = "DDQN"  # DDQN or Reinforce or AC or Keyboard

if __name__ == "__main__":

    if agent_type == "Reinforce":
        env = PlayerV2(render=False, max_frames=500 * config.FPS)
        agent = ReinforceAgentV2(
            input_size=env.num_observations(),
            possible_actions=env.get_actions()
        )
        agent.load("agents/agent_zoo/dfp5")

    if agent_type == "DDQN":
        env = PlayerQ(render=False)
        model_name = "dfq5_epsexp"
        # 尝试从 agent_zoo 加载
        model_path = Path("agents/agent_zoo") / model_name / model_name
        if not model_path.exists():
            # 备选: 直接使用旧路径
            model_path = Path("agents/agent_zoo") / model_name
        load_path = str(model_path)
        
        # 自动选择设备：有GPU用GPU，没有用CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Allowlist QNetwork for safe unpickling and load the full object.
        if hasattr(torch.serialization, "safe_globals"):
            from torch.serialization import safe_globals as _safe_globals

            with _safe_globals([QNetwork]):
                agent = torch.load(load_path, weights_only=False, map_location=device)
        else:
            # add_safe_globals exists in some versions
            torch.serialization.add_safe_globals([QNetwork])
            agent = torch.load(load_path, weights_only=False, map_location=device)
        # 确保模型的device属性与实际设备一致
        agent.device = device

    if agent_type == "AC":
        env = Trainer(render=False, max_frames=500 * config.FPS)
        agent = PPOAgent(
            input_size=env.num_observations(),
            possible_actions=list(range(env.num_actions()))
        )
        # 尝试从 agent_zoo 加载
        model_name = "ppo_vec_agent"
        model_path = Path("agents/agent_zoo") / model_name / f"{model_name}.pth"
        if not model_path.exists():
            # 备选: 直接使用旧路径
            model_path = Path("ppo_vec_agent.pth")
        agent.load(str(model_path))

    if agent_type == "Keyboard":
        env = PlayerV2(render=True, max_frames=500 * config.FPS)
        agent = KeyboardAgent()
    env.play(agent)
    render_info = env.get_render_info()
    render(render_info)
