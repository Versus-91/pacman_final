import pygame
from pygame.locals import *
from vector import Vectors
#from constants import *
from random import randint
from behavior import ModeController



cellw = 16   #TILEWIDTH
cellh= 16   #TILEHEIGHT
row = 36   #NROWS
col = 28  #NCOLS
w = col*cellw #SCREENWIDTH
h =row * cellh #SCREENHEIGHT = NROWS*TILEHEIGHT
screen = (w,h)
#SCREENSIZE = (SCREENWIDTH, SCREENHEIGHT)

stop = 0
up = 1 
down =-1 
left = 2 
right =-2 

pacman=0
SCATTER = 0
CHASE = 1
FREIGHT = 2
SPAWN = 3
powerdot=4

blinky = 4
pinky = 5
inky = 6
clyde = 7

gold = 99
bomb = 98
teleport = 97


gold_img=pygame.transform.scale(pygame.image.load(f'assets/gold.png'),(1.5*cellw,1.5*cellw))
bomb_img=pygame.transform.scale(pygame.image.load(f'assets/bomb.png'),(1.5*cellw,1.5*cellw))
teleport_img=pygame.transform.scale(pygame.image.load(f'assets/teleport.png'),(1.5*cellw,1.5*cellw))

class Award(object):
    def __init__(self, node, level=0):
        self.node = node
        self.name = None
        self.color = None
        self.radius = int(8 * cellw / 16)
        self.collision_distance = int(4 * cellh / 16)
        self.lifespan = 6
        self.timer = 0
        self.destroy = False
        self.visible = True
        #self.point = 500 + level*100
        self.point = None
        self.middleOfNodes(right)
        #self.position = self.node.position
        self.image = None
    
    def update(self, time):
        self.timer += time
        if self.timer >= self.lifespan:
            self.destroy = True

    def middleOfNodes(self, direction):
        if self.node.neighbors[direction] is not None:
            self.target = self.node.neighbors[direction]
            self.position = (self.node.position + self.target.position) / 2.0

    def reset(self):
        self.visible = True

    def render(self, screen):
        if self.visible:
            p = self.position.asInt()
            t = cellh/4
            screen.blit(self.image,(p[0]-t,p[1]-t))



class Gold(Award):
    def __init__(self, node, level=0):
        Award.__init__(self, node, level=0)
        self.name = gold
        self.color = 'yellow'
        self.point = 200 + level*200
        self.image = gold_img

class Bomb(Award):
    def __init__(self, node, level=0):
        Award.__init__(self, node, level=0)
        self.name = bomb
        self.color = 'red'
        #self.point = 200
        self.point = 0
        self.image = bomb_img

class Teleport(Award):
    def __init__(self, node, level=0):
        Award.__init__(self, node, level=0)
        self.name = teleport
        self.color = 'green'
        #self.point = 200
        self.point = 0
        self.image = teleport_img

