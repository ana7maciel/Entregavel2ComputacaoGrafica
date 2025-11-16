import glfw
from OpenGL.GL import *
import numpy as np
from PIL import Image
import ctypes
import math

from fpscounter import FPSCounter

#criação dos shaders no mesmo código pra não dar erro de localização
#vertex shader recebe posição e UV, aplica MVP e repassa UV
VERT_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aUV;

out vec2 uv;
uniform mat4 MVP;

void main() {
    gl_Position = MVP * vec4(aPos, 1.0);
    uv = aUV;
}
"""
#fragment shader aplica convolução (kernel 3x3) e modo cinza opcional
FRAG_SHADER = """
#version 330 core
in vec2 uv;
out vec4 FragColor;

uniform sampler2D tex;
uniform float kernel[9];
uniform bool gray;

void main() {
    float o = 1.0 / 300.0;

    vec2 offs[9] = vec2[](
        vec2(-o,  o), vec2(0,  o), vec2( o,  o),
        vec2(-o,  0), vec2(0,  0), vec2( o,  0),
        vec2(-o, -o), vec2(0, -o), vec2( o, -o)
    );

    vec3 sum = vec3(0);
    for (int i = 0; i < 9; i++)
        sum += texture(tex, uv + offs[i]).rgb * kernel[i];

    if (gray) {
        float c = dot(sum, vec3(0.299, 0.587, 0.114));
        FragColor = vec4(vec3(c), 1.0);
    } else {
        FragColor = vec4(sum, 1.0);
    }
}
"""

#funções auxiliares
#compila shaders
def compileShader(shaderType, source):
    shader = glCreateShader(shaderType)
    glShaderSource(shader, source)
    glCompileShader(shader)

    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader
#carrega textura usando PIL
def loadTexture(path):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
    width, height = img.size
    data = img.tobytes()

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, data)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    return tex

#matrizes de transformação
#projeção perspectiva
def perspective(fov, aspect, near, far):
    f = 1 / math.tan(math.radians(fov) / 2)

    M = np.zeros((4, 4), np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = -1
    M[3, 2] = (2 * far * near) / (near - far)

    return M
#ponto de vista da câmera
def lookAt(pos, target, up):
    f = target - pos
    f /= np.linalg.norm(f)

    s = np.cross(f, up)
    s /= np.linalg.norm(s)

    u = np.cross(s, f)

    M = np.identity(4, np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -np.dot(M[:3, :3], pos)

    return M
#apenas translação (modelo)
def translate(x, y, z):
    M = np.identity(4, np.float32)
    M[3, :3] = [x, y, z]
    return M

#estado da aplicação
class State:
    def __init__(self):
        self.pos = np.array([0, 0, 3], np.float32) #posição da câmera
        self.yaw = -90 #rotação horizontal
        self.pitch = 0 #rotação vertical
        self.speed = 0.12 #velocidade de movimento

        self.gray = False #modo cinza
        #kernels usados na convolução (1–5)
        self.kernels = {
            1: np.array([1,2,1,2,4,2,1,2,1], np.float32) / 16.0,  #blur
            2: np.array([1,1,1,1,-8,1,1,1,1], np.float32),        #bordas
            3: np.array([0,-1,0,-1,5,-1,0,-1,0], np.float32),     #sharpen
            4: np.array([-2,-1,0,-1,1,1,0,1,2], np.float32),      #emboss
            5: np.array([-1,-1,-1,-1,8,-1,-1,-1,-1], np.float32)  #outline
        }

        self.kernel = self.kernels[1]  #filtro inicial
    #direção da câmera baseada em yaw/pitch
    def front(self):
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)

        return np.array([
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
            math.sin(yaw) * math.cos(pitch)
        ], np.float32)

#callbacks teclado e mouse
def keyCallback(win, key, scancode, action, mods):
    if action not in (glfw.PRESS, glfw.REPEAT):
        return

    state = glfw.get_window_user_pointer(win)
    #movimentação
    forward = state.front()
    right = np.cross(forward, np.array([0, 1, 0], np.float32))
    right /= np.linalg.norm(right)
    
    if key == glfw.KEY_W: state.pos += forward * state.speed
    if key == glfw.KEY_S: state.pos -= forward * state.speed
    if key == glfw.KEY_A: state.pos -= right * state.speed
    if key == glfw.KEY_D: state.pos += right * state.speed
    if key == glfw.KEY_SPACE: state.pos[1] -= state.speed
    if key == glfw.KEY_LEFT_SHIFT: state.pos[1] += state.speed
    #rotação da câmera
    if key == glfw.KEY_LEFT: state.yaw -= 2
    if key == glfw.KEY_RIGHT: state.yaw += 2
    if key == glfw.KEY_UP: state.pitch += 2
    if key == glfw.KEY_DOWN: state.pitch -= 2
    #troca de kernel
    mapping = {
        glfw.KEY_1: 1,
        glfw.KEY_2: 2,
        glfw.KEY_3: 3,
        glfw.KEY_4: 4,
        glfw.KEY_5: 5
    }

    if key in mapping:
        state.kernel = state.kernels[mapping[key]]

# Necessário armazenar o caminho para o reset
texture_path = "templo.png"
tex = None

def mouseButtonCallback(win, button, action, mods):
    global tex
    state = glfw.get_window_user_pointer(win)

    #verificação do modo cinza (toggle)
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        state.gray = not state.gray

    #reset da imagem
    if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
        state.pos[:] = [0, 0, 3]
        state.yaw = -90
        state.pitch = 0
        state.gray = False

        # recarrega textura original
        tex = loadTexture(texture_path)

        # kernel identidade
        identity_kernel = np.array([0,0,0,0,1,0,0,0,0], np.float32)
        state.kernel = identity_kernel

#loop principal
def main():
    global tex

    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    win = glfw.create_window(800, 600, "3D + Convolução + Extras", None, None)
    glfw.make_context_current(win)

    #VSync
    glfw.swap_interval(1)

    state = State()
    glfw.set_window_user_pointer(win, state)
    glfw.set_key_callback(win, keyCallback)
    glfw.set_mouse_button_callback(win, mouseButtonCallback)

    glEnable(GL_DEPTH_TEST)
    #2 triângulos com posição e UV
    quad = np.array([
        -1,  1, 0, 0, 1,
        -1, -1, 0, 0, 0,
         1, -1, 0, 1, 0,
         1, -1, 0, 1, 0,
         1,  1, 0, 1, 1,
        -1,  1, 0, 0, 1,
    ], np.float32)
    #criação de VAO e VBO
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)

    stride = 5 * quad.itemsize
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    #criação shader program
    vs = compileShader(GL_VERTEX_SHADER, VERT_SHADER)
    fs = compileShader(GL_FRAGMENT_SHADER, FRAG_SHADER)

    programId = glCreateProgram()
    glAttachShader(programId, vs)
    glAttachShader(programId, fs)
    glLinkProgram(programId)
    glUseProgram(programId)
    #variáveis uniforms
    locMvp = glGetUniformLocation(programId, "MVP")
    locKernel = glGetUniformLocation(programId, "kernel")
    locGray = glGetUniformLocation(programId, "gray")
    #fix do sampler2D - ligando textura na unidade 0
    locTex = glGetUniformLocation(programId, "tex")
    glUniform1i(locTex, 0)
    glActiveTexture(GL_TEXTURE0)
    # ------------------------------------------------------

    tex = loadTexture(texture_path)

    #fps counter
    winWidth, winHeight = glfw.get_window_size(win)

    fps_counter = FPSCounter(average_over=30, stats_interval=5.0)
    fps_counter.initialize_text_rendering(winWidth, winHeight)
    fps_counter.enable_stats_printing(True)

    while not glfw.window_should_close(win):

        #update fps
        fps_counter.update()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(programId)
        #matrizes da cena
        proj = perspective(45, 800 / 600, 0.1, 100)
        view = lookAt(state.pos, state.pos + state.front(), np.array([0, 1, 0], np.float32))
        model = translate(0, 0, -3)

        MVP = model @ view @ proj

        glUniformMatrix4fv(locMvp, 1, GL_FALSE, MVP)
        glUniform1fv(locKernel, 9, state.kernel)
        glUniform1i(locGray, int(state.gray))
        #renderização do quadrado com textura e convolução
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        #desenho fps
        fps_counter.render_fps(x=10, y=winHeight - 30,
                               size=2.0, color=(1.0, 1.0, 0.0))

        glfw.swap_buffers(win)
        glfw.poll_events()

    #limpeza
    glBindVertexArray(0)
    glBindTexture(GL_TEXTURE_2D, 0)
    glUseProgram(0)

    glDeleteShader(vs)
    glDeleteShader(fs)

    glDeleteProgram(programId)

    glDeleteBuffers(1, [vbo])
    glDeleteVertexArrays(1, [vao])

    glDeleteTextures([tex])

    fps_counter.cleanup()
    glfw.terminate()

if __name__ == "__main__":
    main()
