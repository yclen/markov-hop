import sys, io, math, random
sys.stderr = io.StringIO()
import pygame
sys.stderr = sys.__stderr__


def run_simulation(kh, n_mediators, on_p=None, on_q=None):
    # --- Setup ---
    pygame.init()
    WIDTH, HEIGHT = (n_mediators * 120 + 300), 300
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Linked Molecule System")
    icon = pygame.image.load("molecule-icon.png")
    pygame.display.set_icon(icon)
    clock = pygame.time.Clock()

    # --- Config ---
    RADIUS        = 30
    BAR_LENGTH    = 120
    P             = kh / 60

    # --- Colors ---
    BG         = (30,  30,  30)
    GRAY       = (120, 120, 120)
    SENSITIZER = (220, 50,  50)
    MEDIATOR   = (50,  0,   170)
    EMITTER    = (50,  200, 80)
    WHITE      = (240, 240, 240)

    BAR_THICKNESS = 8
    S_X = 100
    CY  = HEIGHT // 2

    # --- SpinningTriangle class ---
    class SpinningTriangle:
        def __init__(self, x, y, radius=30, color=(240, 200, 50), speed=90*14):
            self.x      = x
            self.y      = y
            self.radius = radius
            self.color  = color
            self.speed  = speed
            self.angle  = 0.0

        def update(self, dt):
            self.angle += self.speed * dt

        def draw(self, surface):
            pts = []
            for i in range(3):
                a = math.radians(self.angle + i * 120)
                pts.append((self.x + self.radius * math.cos(a),
                             self.y + self.radius * math.sin(a)))
            pygame.draw.polygon(surface, self.color, pts)

    # --- Build molecule list ---
    def build_molecules():
        mols = []
        mols.append((S_X, CY, SENSITIZER, "Sensitizer"))
        for i in range(n_mediators):
            x = S_X + BAR_LENGTH * (i + 1)
            label = "Mediator" if n_mediators == 1 else f"Mediator {i+1}"
            mols.append((x, CY, MEDIATOR, label))
        x = S_X + BAR_LENGTH * (n_mediators + 1)
        mols.append((x, CY, EMITTER, "Emitter"))
        return mols

    molecules  = build_molecules()
    font       = pygame.font.SysFont("Arial", 16)
    font_small = pygame.font.SysFont("Arial", 13)

    # --- Spinning triangle ---
    sensitizer_tri = SpinningTriangle(S_X, CY)
    tri_index = 0

    # --- Timer / run state ---
    runs         = []
    current_run  = None
    run_start    = None
    timer_running = False

    def start_new_run():
        nonlocal current_run, run_start, timer_running
        runs.append({"label": "Run", "elapsed": 0.0, "done": False})
        current_run   = len(runs) - 1
        run_start     = pygame.time.get_ticks()
        timer_running = True

    def reset_triplet():
        nonlocal tri_index, run_start, timer_running
        if current_run is not None and timer_running:
            elapsed = (pygame.time.get_ticks() - run_start) / 1000.0
            runs[current_run]["elapsed"] = elapsed
            runs[current_run]["done"]    = True
            timer_running = False
        tri_index = 0
        sensitizer_tri.x, sensitizer_tri.y = molecules[0][0], molecules[0][1]
        start_new_run()

    start_new_run()

    # --- Main loop ---
    while True:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if on_q is not None:
                    on_q()
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_triplet()
                if event.key == pygame.K_p and on_p is not None:
                    on_p()
                if event.key == pygame.K_q:
                    if on_q is not None:
                        on_q()
                    pygame.quit()
                    return

        screen.fill(BG)

        # Bars
        for i in range(len(molecules) - 1):
            pygame.draw.line(screen, GRAY,
                             (molecules[i][0],   molecules[i][1]),
                             (molecules[i+1][0], molecules[i+1][1]),
                             BAR_THICKNESS)

        # Molecules + labels
        for x, y, color, label in molecules:
            pygame.draw.circle(screen, color, (x, y), RADIUS)
            text = font.render(label, True, WHITE)
            screen.blit(text, (x - text.get_width() // 2, y + RADIUS + 8))

        # Hopping logic
        last_index = len(molecules) - 1
        if tri_index != last_index and random.random() < P:
            if random.random() < 0.5:
                tri_index = min(tri_index + 1, last_index)
            else:
                tri_index = max(tri_index - 1, 1)
            sensitizer_tri.x, sensitizer_tri.y = molecules[tri_index][0], molecules[tri_index][1]

            if tri_index == last_index and timer_running:
                elapsed = (pygame.time.get_ticks() - run_start) / 1000.0
                runs[current_run]["elapsed"] = elapsed
                runs[current_run]["done"]    = True
                timer_running = False

        # Update live elapsed
        if timer_running and current_run is not None:
            runs[current_run]["elapsed"] = (pygame.time.get_ticks() - run_start) / 1000.0

        # Spinning triangle
        sensitizer_tri.update(dt)
        sensitizer_tri.draw(screen)

        padding = 10

        # --- Draw parameters (top-right) ---
        params = [
            f"N_mediators: {n_mediators}",
            f"k_hop: {kh} (1/s)",
            f"t_hop: {1/kh:.3f} (s)",
        ]
        for i, p in enumerate(params):
            t = font_small.render(p, True, WHITE)
            screen.blit(t, (WIDTH - t.get_width() - padding, padding + i * 18))

        # --- Draw run timers (top-left, horizontal) ---
        x_offset = padding
        for run in runs:
            label_str = f"Run: {run['elapsed']:.2f}s"
            t = font_small.render(label_str, True, WHITE)
            screen.blit(t, (x_offset, padding))
            x_offset += t.get_width() + 20

        pygame.display.flip()


if __name__ == "__main__":
    run_simulation(kh=5, n_mediators=7)