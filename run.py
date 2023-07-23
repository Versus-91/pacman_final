cellw = 16  # TILEWIDTH
cellh = 16  # TILEHEIGHT
row = 36  # NROWS
col = 28  # NCOLS
w = col * cellw  # SCREENWIDTH
h = row * cellh  # SCREENHEIGHT = NROWS*TILEHEIGHT
screen = (w, h)
# SCREENSIZE = (SCREENWIDTH, SCREENHEIGHT)
# screen_color = (0, 0, 0)

# pacman_col = (255, 255, 0)

stop = 0  # STOP = 0
up = 1  # UP = 1 #
down = -1  # DOWN = -1
left = 2  # LEFT = 2
right = -2  # RIGHT = -2


scattering_mode = 0
chasing_mode = 1
scared_mode = 2
respawning_mode = 3


pacman = 0  # PACMAN = 0
SCATTER = 0
CHASE = 1
FREIGHT = 2
SPAWN = 3
powerdot = 4

blinky = 4
pinky = 15
inky = 6
clyde = 7

sue = 16
funky = 17

gold = 99
bomb = 98
shield = 97

import numpy as np
import pygame
from pygame.locals import *
from bfs import minDistance

# from pacman import mypacman
from nodes import NodeGroup, PelletGroup

# from ghost import GhostGroup
from vector import Vectors
from awards import Gold, Bomb, Shield


from objects import Pacman, GhostGroup

from button import Button

player_images = []
for i in range(1, 5):
    player_images.append(
        pygame.transform.scale(
            pygame.image.load(f"assets/player_images/{i}.png"), (cellw / 2, cellw / 2)
        )
    )


class GameState:
    def __init__(self):
        self.lives = 0
        self.frame = []
        self.state = []
        self.invalid_move = False
        self.total_pellets = 0
        self.collected_pellets = 0
        self.food_distance = -1
        self.powerup_distance = -1
        self.ghost_distance = -1
        self.scared_ghost_distance = -1
        self.image = []
        self.x = 0
        self.y = 0


# Music/Sounds

pygame.mixer.init()
music_start = pygame.mixer.music.load("assets/pacman_beginning.wav")
# music_start = pygame.mixer.Sound("assets/pacman_beginning.wav")
eatghost_sound = pygame.mixer.Sound("assets/pacman_eatghost.wav")
eatdot_sound = pygame.mixer.Sound("assets/pacman_chomp.wav")
death_sound = pygame.mixer.Sound("assets/pacman_death.wav")
powerdot_sound = pygame.mixer.Sound("assets/pacman_intermission.wav")


class GameController(object):
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode(screen, 0, 32)
        self.background = None
        self.clock = pygame.time.Clock()
        self.lives = 3
        self.level = 0
        self.won = False
        self.lost = False
        self.score = 0
        self.counter = 0
        self.level_map = f"levels/{self.level}.txt"
        self.level_up = False
        self.timer = 0

        self.gold = None
        self.gold_node = None
        self.bomb = None
        self.bomb_node = None
        self.shield = None
        self.shield_node = None

        self.font = pygame.font.Font("freesansbold.ttf", 20)
        self.newfont = pygame.font.Font("assets/font.ttf", 50)
        self.otherfont = pygame.font.Font("assets/font.ttf", 40)

    def show_levels(self):
        while True:
            self.screen.fill("black")
            mouse = pygame.mouse.get_pos()
            button1 = Button(
                image=pygame.image.load("assets/Play Rect.png"),
                pos=(w // 2, 90),
                text_input="LEVEL 1",
                font=pygame.font.Font("assets/font.ttf", 45),
                base_color="blue",
                hovering_color="White",
            )
            button2 = Button(
                image=pygame.image.load("assets/Options Rect.png"),
                pos=(w // 2, 245),
                text_input="LEVEL 2",
                font=pygame.font.Font("assets/font.ttf", 45),
                base_color="red",
                hovering_color="White",
            )
            button3 = Button(
                image=pygame.image.load("assets/Options Rect.png"),
                pos=(w // 2, 390),
                text_input="LEVEL 3",
                font=pygame.font.Font("assets/font.ttf", 45),
                base_color="purple",
                hovering_color="White",
            )
            button4 = Button(
                image=pygame.image.load("assets/Back Rect.png"),
                pos=(w // 2, 525),
                text_input="BACK",
                font=pygame.font.Font("assets/font.ttf", 20),
                base_color="red",
                hovering_color="White",
            )
            for button in [button1, button2, button3, button4]:
                button.changeColor(mouse)
                button.update(self.screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                    pygame.quit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if button1.checkForInput(mouse):
                        self.startGame()

                        while True:
                            self.update()

                    if button2.checkForInput(mouse):
                        self.level = 1
                        # self.startGame()

                        self.nextLevel()
                        while True:
                            self.update()
                    if button3.checkForInput(mouse):
                        self.level = 2
                        # self.startGame()

                        self.nextLevel()
                        while True:
                            self.update()
                    if button4.checkForInput(mouse):
                        self.startMenu()

            pygame.display.update()

    def draw_misc(self):
        score_text = self.font.render(f"Score:{self.score}", True, "white")
        self.screen.blit(score_text, (10, h - 20))
        for ghost in self.ghosts:
            if ghost.mode.current_mode == FREIGHT:
                pygame.draw.circle(
                    self.screen, "yellow", (140, h - 15), cellw * 2 / 3
                )  # powerup is active
        # print("powerup")
        for i in range(self.lives):
            self.screen.blit(
                pygame.transform.scale(player_images[0], (cellw, cellh)),
                (w - 180 + i * 40, h - 20),
            )
        if self.lost:
            self.screen.fill("black")
            text = self.otherfont.render("You LOST", True, "red")
            menu = text.get_rect(center=(w // 2, h // 2))
            self.screen.blit(text, menu)
            text1 = self.font.render("Press SPACE to return to MENU", True, "white")
            menu1 = text.get_rect(center=(w // 2 - 20, h - 25))
            self.screen.blit(text1, menu1)
            # print("lost")
        if self.level_up:
            self.screen.fill("black")
            text = self.otherfont.render("NEXT LEVEL", True, "white")
            menu = text.get_rect(center=(w // 2, h // 2))
            self.screen.blit(text, menu)
            text1 = self.font.render("PRESS SPACE TO CONTINUE", True, "white")
            menu1 = text.get_rect(center=(w // 2 + 60, h - 25))
            self.screen.blit(text1, menu1)
        if self.won:
            self.screen.fill("black")
            text = self.otherfont.render("YOU WON", True, "green")
            menu = text.get_rect(center=(w // 2, h // 2))
            self.screen.blit(text, menu)
            text1 = self.font.render("Press SPACE to return to MENU", True, "white")
            menu1 = text.get_rect(center=(w // 2 - 20, h - 25))
            self.screen.blit(text1, menu1)
            # print("won")

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def end(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def restartGame(self):
        self.won = False
        self.lost = False
        self.lives = 3
        self.level = 0
        self.level_map = f"levels/{self.level}.txt"
        self.score = 0

        self.startGame()

    def nextLevel(self):
        pygame.mixer.music.play(-1)
        self.gold = None
        self.bomb = None
        self.shield = None
        self.level_up = False
        self.level_map = f"levels/{self.level}.txt"
        self.nodes = NodeGroup(self.level_map)
        if self.level == 1:
            self.nodes.default_color = "red"
        if self.level == 2:
            self.nodes.default_color = "purple"
        timer = 0

        self.nodes.setPortalPair((0, 17), (27, 17))  # change numbers to letters
        box = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(box, (12, 14), left)
        self.nodes.connectHomeNodes(box, (15, 14), right)
        self.pellets = PelletGroup(self.level_map)
        # self.pacman = mypacman(self.nodes.getNodeFromTiles(15, 26))
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman, self.level)

        self.ghosts.blinky.initialNode(self.nodes.getNodeFromTiles(2 + 11.5, 3 + 14))
        self.ghosts.inky.initialNode(self.nodes.getNodeFromTiles(0 + 11.5, 3 + 14))
        self.ghosts.sue.initialNode(self.nodes.getNodeFromTiles(2 + 11.5, 3 + 14))
        if self.level == 1:
            self.ghosts.clyde.initialNode(self.nodes.getNodeFromTiles(4 + 11.5, 3 + 14))
        self.ghosts.sue.initialNode(self.nodes.getNodeFromTiles(4 + 11.5, 3 + 14))
        if self.level == 2:
            self.ghosts.funky.initialNode(self.nodes.getNodeFromTiles(2 + 11.5, 3 + 14))

        self.eatenPellets = []
        self.gold_initial_position = (9, 20)
        self.bomb_initial_position = (9, 14)
        self.shield_initial_position = (15, 26)

        self.ghosts.setRespawnTarget(self.nodes.getNodeFromTiles(2 + 11.5, 3 + 14))

        self.nodes.denyHomeAccess(self.pacman)  #!
        # self.nodes.denyHomeAccess(self.ghosts.sue)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2 + 11.5, 3 + 14, left, self.ghosts)
        self.nodes.denyAccessList(2 + 11.5, 3 + 14, right, self.ghosts)

        self.nodes.denyAccessList(12, 14, up, self.ghosts)
        self.nodes.denyAccessList(15, 14, up, self.ghosts)
        self.nodes.denyAccessList(12, 26, up, self.ghosts)
        self.nodes.denyAccessList(15, 26, up, self.ghosts)

        # self.ghosts.blinky.gets_angry(self.counter)

    def resetLevel(self, level):
        self.gold = None
        self.bomb = None
        self.shield = None

        self.level = level
        # self.pacman = mypacman(self.nodes.getNodeFromTiles(15, 26))
        self.pacman.reset()
        self.ghosts.reset()

    def setBackground(self):
        self.background = pygame.surface.Surface(screen).convert()
        self.background.fill("black")

    def startGame(self):
        # music_start.play()
        # pygame.mixer.music.play(-1)
        self.setBackground()

        self.nodes = NodeGroup(self.level_map)
        self.nodes.setPortalPair((0, 17), (27, 17))  # change numbers to letters
        box = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(box, (12, 14), left)
        self.nodes.connectHomeNodes(box, (15, 14), right)
        self.pellets = PelletGroup(self.level_map)

        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))

        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)

        self.ghosts.blinky.initialNode(self.nodes.getNodeFromTiles(2 + 11.5, 0 + 14))
        self.ghosts.pinky.initialNode(self.nodes.getNodeFromTiles(2 + 11.5, 3 + 14))
        self.ghosts.inky.initialNode(self.nodes.getNodeFromTiles(0 + 11.5, 3 + 14))
        self.ghosts.clyde.initialNode(self.nodes.getNodeFromTiles(4 + 11.5, 3 + 14))
        self.eatenPellets = []

        self.gold_initial_position = (9, 20)
        self.bomb_initial_position = (9, 14)
        self.shield_initial_position = (15, 26)

        self.ghosts.setRespawnTarget(self.nodes.getNodeFromTiles(2 + 11.5, 3 + 14))

        self.nodes.denyHomeAccess(self.pacman)  #!

        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2 + 11.5, 3 + 14, left, self.ghosts)
        self.nodes.denyAccessList(2 + 11.5, 3 + 14, right, self.ghosts)

        self.nodes.denyAccessList(12, 14, up, self.ghosts)
        self.nodes.denyAccessList(15, 14, up, self.ghosts)
        self.nodes.denyAccessList(12, 26, up, self.ghosts)
        self.nodes.denyAccessList(15, 26, up, self.ghosts)

        self.ghosts.inky.startNode.denyAccess(right, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(left, self.ghosts.clyde)

    def updateScore(self, points):
        self.score += points

    def close(self):
        exit()

    def update(self):  # remove time later !
        delta_t = self.clock.tick(120) / 1000.0
        if self.counter < 19:  # spped of eating my pacman
            self.counter += 1
        else:
            self.counter = 0

        self.pacman.update(delta_t)  # remove time?
        self.pellets.update(delta_t)
        self.checkEvents()
        self.eatDots()

        if self.gold is not None:
            self.gold.update(delta_t)
        self.gettingGold()

        if self.bomb is not None:
            self.bomb.update(delta_t)
        self.gettingBomb()

        # if self.shield is not None:
        #     self.shield.update(delta_t)
        # self.gettingShield()

        self.ghosts.update(delta_t)
        self.checkGhostEvents()
        self.render()
        self.get_frame()

        # print(self.level, "level")
        # print(self.ghosts.blinky.mode.current_mode,self.ghosts.inky.mode.current_mode,self.ghosts.pinky.mode.current_mode,self.ghosts.clyde.mode.current_mode)
        # print(self.pacman.position==game.ghosts.blinky.pacman.position,game.pacman.alive)

    def get_frame(self):
        raw_maze_data = []
        with open(self.level_map, "r") as f:
            for line in f:
                raw_maze_data.append(line.split())
        raw_maze_data = np.array(raw_maze_data)
        self.state = np.zeros(raw_maze_data.shape)
        for idx, values in enumerate(raw_maze_data):
            for id, value in enumerate(values):
                if value in ["9", "=", "X", "3", "4", "5", "6", "7", "8"]:
                    self.state[idx][id] = 1
        # for idx, pellet in enumerate(self.eatenPellets):
        #     x = int(pellet.position.x / 16)
        #     y = int(pellet.position.y / 16)
        #     self.state[y][x] = 2
        for idx, pellet in enumerate(self.pellets.pelletList):
            x = int(pellet.position.x / 16)
            y = int(pellet.position.y / 16)
            if pellet.name == 3:
                self.state[y][x] = 3
            else:
                self.state[y][x] = 4
        pacman_x = int(round(self.pacman.position.x / 16))
        pacman_y = int(round(self.pacman.position.y / 16))
        self.state[pacman_y][pacman_x] = 5
        # assert self.state[y][x] != 1
        for ghost in enumerate(self.ghosts):
            x = int(round(ghost[1].position.x / 16))
            y = int(round(ghost[1].position.y / 16))
            if (
                ghost[1].mode.current_mode is not scared_mode
                and ghost[1].mode.current_mode is not respawning_mode
            ):
                self.state[y][x] = -6
            elif ghost[1].mode.current_mode is scared_mode:
                if self.state[y][x] != 5:
                    self.state[y][x] = 6
        if self.bomb:
            x = int(round(self.bomb.position.x / 16))
            y = int(round(self.bomb.position.y / 16))
            if self.state[y][x] != 5:
                self.state[y][x] = 7
        # dist = math.sqrt((self.pacman_prev.x - x)**2 + (self.pacman_prev.y - x)**2)
        # if abs(self.pacman_prev.x - x) >= 16 or abs(self.pacman_prev.y - y) >= 16:
        #     self.pacman_prev = self.pacman.position
        #     print("move",self.pacman.position)

        return self.state[3:34, :]

    def perform_action(self, action):
        state = None
        invalid_move = False
        info = GameState()
        lives = self.lives
        info.frame = self.get_frame()
        info.image = pygame.surfarray.array3d(pygame.display.get_surface())
        # info.state = self.get_state()
        if not self.pacman.validDirection(action):
            invalid_move = True
        delta_t = self.clock.tick(120) / 1000.0
        if self.counter < 19:
            self.counter += 1
        else:
            self.counter = 0
        self.pacman.update(delta_t, action)
        self.pellets.update(delta_t)
        self.checkEvents()
        self.eatDots()
        if self.gold is not None:
            self.gold.update(delta_t)
        self.gettingGold()
        if self.bomb is not None:
            self.bomb.update(delta_t)
        self.gettingBomb()
        # if self.shield is not None:
        #     self.shield.update(delta_t)
        # self.gettingShield()
        self.ghosts.update(delta_t)
        self.checkGhostEvents()
        self.render()
        if lives == self.lives:
            info.frame = self.get_frame()
            # info.state = self.get_state()
            info.image = pygame.surfarray.array3d(pygame.display.get_surface())
        info.lives = self.lives
        row_indices, _ = np.where(info.frame == 5)
        info.invalid_move = invalid_move
        info.total_pellets = len(self.pellets.pelletList) + len(self.eatenPellets)
        info.collected_pellets = len(self.eatenPellets)
        if row_indices.size > 0:
            info.food_distance = minDistance(info.frame, 5, 3, [-6, 1])
            info.powerup_distance = minDistance(info.frame, 5, 4, [-6, 1])
            info.ghost_distance = minDistance(info.frame, 5, -6)
            info.scared_ghost_distance = minDistance(info.frame, 5, 6)
        return ([], self.score, (self.lives == 0 or self.pellets.isEmpty()), info)

    def eatDots(self):
        dot = self.pacman.collectObjectives(self.pellets.pelletList)
        # dot = self.pacman.eatDots(self.pellets.pelletList)
        if dot:
            # eatdot_sound.play()
            self.pellets.numEaten += 1
            self.updateScore(dot.points)
            self.pellets.pelletList.remove(dot)
            if self.pellets.numEaten == 20:
                self.ghosts.inky.startNode.allowAccess(right, self.ghosts.inky)
            if self.pellets.numEaten == 60:
                self.ghosts.clyde.startNode.allowAccess(left, self.ghosts.clyde)
            # print("remain dots",len(self.pellets.pelletList))
            if len(self.pellets.pelletList) < 5:
                self.ghosts.blinky.gets_angry(self.counter)
            if dot.name == powerdot:
                self.ghosts.startScaring()

            if self.pellets.isEmpty():
                # if len(self.pellets.pelletList) < 235:
                # self.won=True
                self.level += 1
                self.end()
                if self.level < 3:
                    self.level_up = True
                else:
                    self.restartGame()
                    # self.won=True

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current_mode is FREIGHT:
                    # eatghost_sound.play()
                    # ghost.visible = False
                    # self.ghosts.pinky.visible=False
                    self.updateScore(ghost.point)
                    self.ghosts.upgradePoint()
                    self.nodes.allowHomeAccess(ghost)
                    ghost.startRespawning()
                elif ghost.mode.current_mode is not SPAWN:
                    if self.pacman.alive:
                        # death_sound.play()
                        self.lives -= 1
                        self.pacman.die()
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.lost = True
                            # self.restartGame()
                        else:
                            self.resetLevel(self.level)

    def gettingGold(self):
        if self.pellets.numEaten == 30:
            if self.gold is None:
                self.gold = Gold(self.nodes.getNodeFromTiles(9, 20), self.level)
        if self.gold is not None:
            if self.pacman.collideCheck(self.gold):
                self.updateScore(self.gold.point)
                self.gold = None
            elif self.gold.destroy:
                self.gold = None

    def nearestGhost(self):
        nearest_ghost = None
        min_distance = 600.0
        for ghost in self.ghosts:
            dist = Vectors.lenght(self.pacman.position - ghost.position)
            if dist < min_distance:
                nearest_ghost = ghost
                min_distance = dist
        return nearest_ghost

    def gettingBomb(self):
        if self.pellets.numEaten == 50:
            if self.bomb is None:
                self.bomb = Bomb(self.nodes.getNodeFromTiles(9, 14), self.level)

        if self.bomb is not None:
            if self.pacman.collideCheck(self.bomb):
                ghost = self.nearestGhost()
                # print(ghost.color,"there")
                # ghost.visible = False
                # self.updateScore(ghost.points)
                self.nodes.allowHomeAccess(ghost)
                ghost.mode.current_mode = SPAWN
                ghost.startRespawning()

                # self.updateScore(self.bomb.point)
                self.bomb = None
            elif self.bomb.destroy:
                self.bomb = None

    def gettingShield(self):
        if self.pellets.numEaten == 5:
            if self.shield is None:
                self.shield = Shield(self.nodes.getNodeFromTiles(15, 26), self.level)
        if self.shield is not None:
            if self.pacman.collideCheck(self.shield):
                for ghost in self.ghosts:
                    # if (cellw * 1)**2 < (self.pacman.position - ghost.position).magnitudeSquared():

                    if 100 > Vectors.lenght(self.pacman.position - ghost.position):
                        # if 20 < Vectors.lenght(self.pacman.position - ghost.position):
                        # ghost.mode.current_mode=scattering_mode
                        # ghost.scatter()
                        # ghost.goal=self.pacman.position*(-1)
                        ghost.mode.setScaredMode()
                        # ghost.startScaring()
                        print(ghost.name, "name")
                        # ghost.startShield()

                self.shield = None
            elif self.shield.destroy:
                self.shield = None

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
                pygame.quit()
            elif self.level_up and event.type == KEYDOWN:
                if event.key == K_SPACE:
                    self.nextLevel()
            elif (self.won or self.lost) and event.type == KEYDOWN:
                if event.key == K_SPACE:
                    self.restartGame()
                    # here menu
                    # self.startMenu()

    def startMenu(self):
        while True:
            self.screen.fill("black")
            self.setBackground()

            mouse = pygame.mouse.get_pos()

            text = self.newfont.render("MAIN MENU", True, "#b68f40")
            menu = text.get_rect(center=(w // 2, 45))

            play_button = Button(
                image=pygame.image.load("assets/Play Rect.png"),
                pos=(w // 2, 215),
                text_input="PLAY",
                font=pygame.font.Font("assets/font.ttf", 55),
                base_color="orange",
                hovering_color="White",
            )
            levels_button = Button(
                image=pygame.image.load("assets/Options Rect.png"),
                pos=(w // 2, 365),
                text_input="LEVELS",
                font=pygame.font.Font("assets/font.ttf", 55),
                base_color="purple",
                hovering_color="White",
            )
            quit_button = Button(
                image=pygame.image.load("assets/Quit Rect.png"),
                pos=(w // 2, 515),
                text_input="QUIT",
                font=pygame.font.Font("assets/font.ttf", 55),
                base_color="red",
                hovering_color="White",
            )

            self.screen.blit(text, menu)

            for button in [play_button, levels_button, quit_button]:
                button.changeColor(mouse)
                button.update(self.screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                    pygame.quit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if play_button.checkForInput(mouse):
                        self.startGame()
                        while True:
                            self.update()

                    if levels_button.checkForInput(mouse):
                        self.show_levels()
                    if quit_button.checkForInput(mouse):
                        exit()
                        pygame.quit()

            pygame.display.update()

    def render(self):
        self.screen.blit(self.background, (0, 0))
        self.nodes.render(self.screen, self.level_map)
        self.pellets.render(self.screen)
        self.pacman.draw(self.screen, self.counter)
        self.ghosts.render(self.screen)

        if self.gold is not None:
            self.gold.render(self.screen)

        if self.bomb is not None:
            self.bomb.render(self.screen)

        # if self.shield is not None:
        #     self.shield.render(self.screen)

        self.draw_misc()
        pygame.display.update()


if __name__ == "__main__":
    game = GameController()
    pygame.display.set_caption("PAC-MAN")
    # here menu
    # game.startMenu()
    game.startGame()
    while True:
        game.update()
