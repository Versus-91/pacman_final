import pygame


class Ghost:
    def __init__(self, x, y, target, speed, img, direction, dead, box, id, level, cell, powerup, w, eaten_ghosts, screen, poweredup, dead_img):
        self.x = x
        self.screen = screen
        self.eaten_ghosts = eaten_ghosts
        self.level = level
        self.powerup = powerup
        self.poweredup = poweredup
        self.cell = cell
        self.w = w
        self.y = y
        self.centerx = self.x+cell//2
        self.centery = self.y+cell//2
        self.target = target
        self.speed = speed
        self.dead_image = dead_img
        self.img = img
        self.dead = dead
        self.direction = direction
        self.inbox = box
        self.id = id
        self.turns, self.inbox = self.check_collision()
        self.rect = self.draw()

    def draw(self):
        if (not self.powerup and not self.dead) or (self.eaten_ghosts[self.id] and self.powerup and not self.dead):
            self.screen.blit(self.img, (self.x, self.y))
        elif self.powerup and not self.dead and not self.eaten_ghosts[self.id]:
            self.screen.blit(self.poweredup, (self.x, self.y))
        else:
            self.screen.blit(self.dead_image, (self.x, self.y))

        ghos_rect = pygame.rect.Rect(
            (self.centerx-self.cell//2, self.centery-self.cell//2), (self.cell, self.cell))
        return ghos_rect

    def check_collision(self):
        mid = self.cell//2
        col = int(self.centerx//self.cell)
        row = int(self.centery//self.cell)
        self.turns = [False, False, False, False]
        if 0 < self.centerx//self.cell < len(self.level[0])-1:
            # escape the door
            if self.level[int((self.centery-mid)//self.cell)][col] == 9:
                self.turns[2] = True

            if self.level[row][int(self.centerx-mid)//self.cell] < 3 or (self.level[row][int(self.centerx-mid)//self.cell] == 9 and (
                    self.inbox or self.dead)):

                self.turns[1] = True

            if self.level[row][int(self.centerx+mid)//self.cell] < 3 or (self.level[row][int(self.centerx+mid)//self.cell] == 9 and (
                    self.inbox or self.dead)):

                self.turns[0] = True

            if self.level[int(self.centery+mid)//self.cell][col] < 3 or (self.level[int(self.centery+mid)//self.cell][col] == 9 and (
                    self.inbox or self.dead)):

                self.turns[3] = True

            if self.level[int(self.centery-mid)//self.cell][col] < 3 or (self.level[int(self.centery-mid)//self.cell][col] == 9 and (
                    self.inbox or self.dead)):

                self.turns[2] = True

            if self.direction == 2 or self.direction == 3:
                if self.cell//3 <= self.centerx % self.cell <= 2*self.cell//3:
                    if self.level[int(self.centery+mid)//self.cell][col] < 3 or (self.level[int(self.centery+mid)//self.cell][col] == 9 and (
                            self.inbox or self.dead)):

                        self.turns[3] = True

                    if self.level[int(self.centery-mid)//self.cell][col] < 3 or (self.level[int(self.centery-mid)//self.cell][col] == 9 and (
                            self.inbox or self.dead)):

                        self.turns[2] = True
                if self.cell//3 <= self.centery % self.cell <= 2*self.cell//3:
                    if self.level[row][col-1] < 3 or (self.level[row][col-1] == 9 and (
                            self.inbox or self.dead)):

                        self.turns[1] = True

                    if self.level[row][col+1] < 3 or (self.level[row][col+1] == 9 and (
                            self.inbox or self.dead)):

                        self.turns[0] = True

            if self.direction == 0 or self.direction == 1:
                if self.cell//3 <= self.centerx % self.cell <= 2*self.cell//3:
                    if self.level[int(self.centery+mid)//self.cell][col] < 3 or (self.level[int(self.centery+mid)//self.cell][col] == 9 and (
                            self.inbox or self.dead)):

                        self.turns[3] = True

                    if self.level[int(self.centery-mid)//self.cell][col] < 3 or (self.level[int(self.centery-mid)//self.cell][col] == 9 and (
                            self.inbox or self.dead)):

                        self.turns[2] = True
                if self.cell//3 <= self.centery % self.cell <= 2*self.cell//3:
                    if self.level[row][int(self.centerx-mid)//self.cell] < 3 or (self.level[row][int(self.centerx-mid)//self.cell] == 9 and (
                            self.inbox or self.dead)):

                        self.turns[1] = True

                    if self.level[row][int(self.centerx+mid)//self.cell] < 3 or (self.level[row][int(self.centerx+mid)//self.cell] == 9 and (
                            self.inbox or self.dead)):

                        self.turns[0] = True
        else:
            self.turns[0] = True
            self.turns[1] = True

        if 10 < self.x//self.cell < 15 and 5 <= self.y//self.cell < 7:
            # box row and col indices 12-18 14-17
            self.inbox = True
        else:
            self.inbox = False

        return self.turns, self.inbox

    def move_or(self):
        # orange turn for pursuit    go right as a rule
        if self.direction == 0:
            if self.target[0] > self.x and self.turns[0]:
                self.x += self.speed
            elif not self.turns[0]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.y -= self.speed
                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
            elif self.turns[0]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                if self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                else:
                    self.x += self.speed

        elif self.direction == 1:
            if self.target[1] > self.y and self.turns[3]:
                self.direction = 3
            elif self.target[0] < self.x and self.turns[1]:
                self.x -= self.speed
            elif not self.turns[1]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
            elif self.turns[1]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                if self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y += self.speed
                else:
                    self.x -= self.speed

        elif self.direction == 2:
            if self.target[0] < self.x and self.turns[1]:
                self.direction = 1
                self.x -= self.speed
            elif self.target[1] < self.y and self.turns[2]:
                self.direction = 2
                self.y -= self.speed
            elif not self.turns[2]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed

                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
            elif self.turns[2]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                else:
                    self.y -= self.speed
        elif self.direction == 3:
            if self.target[1] > self.y and self.turns[3]:
                self.y += self.speed
            elif not self.turns[3]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
            elif self.turns[3]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                else:
                    self.y += self.speed
        if self.x < -5:
            self.x = self.w-3
        elif self.x > self.w:
            self.x = - 5
        return self.x, self.y, self.direction

    # blinky behavior
    def move_red(self):
        # red avoid wall and go forward

        if self.direction == 0:
            if self.target[0] > self.x and self.turns[0]:
                self.x += self.speed
            elif not self.turns[0]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.y -= self.speed
                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
            elif self.turns[0]:
                self.x += self.speed

        elif self.direction == 1:
            if self.target[0] < self.x and self.turns[1]:
                self.x -= self.speed
            elif not self.turns[1]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
            elif self.turns[1]:
                self.x -= self.speed

        elif self.direction == 2:
            if self.target[1] < self.y and self.turns[2]:
                self.direction = 2
                self.y -= self.speed
            elif not self.turns[2]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed

                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed

                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
            elif self.turns[2]:
                self.y -= self.speed

        elif self.direction == 3:
            if self.target[1] > self.y and self.turns[3]:
                self.y += self.speed
            elif not self.turns[3]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed

                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
            elif self.turns[3]:
                self.y += self.speed
        if self.x < -5:
            self.x = self.w-3
        elif self.x > self.w:
            self.x = - 5
        return self.x, self.y, self.direction

    def move_blue(self):
        # ble turns go up or down to catch pacman , left right to avoid walls
        if self.direction == 0:
            if self.target[0] > self.x and self.turns[0]:
                self.x += self.speed
            elif not self.turns[0]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.y -= self.speed
                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
            elif self.turns[0]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                if self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                else:
                    self.x += self.speed

        elif self.direction == 1:
            if self.target[1] > self.y and self.turns[3]:
                self.direction = 3
            elif self.target[0] < self.x and self.turns[1]:
                self.x -= self.speed
            elif not self.turns[1]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
            elif self.turns[1]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                if self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y += self.speed
                else:
                    self.x -= self.speed

        elif self.direction == 2:
            if self.target[1] < self.y and self.turns[2]:
                self.direction = 2
                self.y -= self.speed
            elif not self.turns[2]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed

                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
            elif self.turns[2]:
                self.y -= self.speed

        elif self.direction == 3:
            if self.target[1] > self.y and self.turns[3]:
                self.y += self.speed
            elif not self.turns[3]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
            elif self.turns[3]:
                self.y += self.speed
        if self.x < -5:
            self.x = self.w-3
        elif self.x > self.w:
            self.x = - 5
        return self.x, self.y, self.direction

    def move_pink(self):  # opposite if blue
        # pink turns left right to catch hero , up down to avoid walls
        if self.direction == 0:
            if self.target[0] > self.x and self.turns[0]:
                self.x += self.speed
            elif not self.turns[0]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.y -= self.speed
                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
            elif self.turns[0]:
                self.x += self.speed

        elif self.direction == 1:
            if self.target[1] > self.y and self.turns[3]:
                self.direction = 3
            elif self.target[0] < self.x and self.turns[1]:
                self.x -= self.speed
            elif not self.turns[1]:
                if self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
            elif self.turns[1]:
                self.x -= self.speed

        elif self.direction == 2:
            if self.target[0] < self.x and self.turns[1]:
                self.direction = 1
                self.x -= self.speed
            elif self.target[1] < self.y and self.turns[2]:
                self.direction = 2
                self.y -= self.speed
            elif not self.turns[2]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.target[1] > self.y and self.turns[3]:
                    self.direction = 3
                    self.y += self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.turns[3]:
                    self.direction = 3
                    self.y += self.speed

                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
            elif self.turns[2]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                else:
                    self.y -= self.speed
        elif self.direction == 3:
            if self.target[1] > self.y and self.turns[3]:
                self.y += self.speed
            elif not self.turns[3]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.target[1] < self.y and self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[2]:
                    self.direction = 2
                    self.y -= self.speed
                elif self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                elif self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
            elif self.turns[3]:
                if self.target[0] > self.x and self.turns[0]:
                    self.direction = 0
                    self.x += self.speed
                elif self.target[0] < self.x and self.turns[1]:
                    self.direction = 1
                    self.x -= self.speed
                else:
                    self.y += self.speed
        if self.x < -5:
            self.x = self.w-3
        elif self.x > self.w:
            self.x = - 5
        return self.x, self.y, self.direction
