# Entregável 2 - Computação Gráfica

Disciplina: Computação Gráfica | Alunos: Ana Karolina Maciel, Carlo e Eduardo Fagundes

Link Repositório: https://github.com/ana7maciel/Entregavel2ComputacaoGrafica.git

Elevator Pitch:
Esta aplicação demonstra, de forma interativa e em tempo real, o uso de filtros de convolução aplicados à imagem de um templo. O usuário pode alternar dinamicamente entre cinco kernels diferentes, ativar o modo em tons de cinza e restaurar a imagem original com apenas um clique.
Além disso, o projeto inclui uma câmera 3D completa, com movimentação em primeira pessoa e rotação do ponto de vista, proporcionando uma experiência imersiva. Para complementar, foram implementados um contador de FPS e V-Sync, garantindo desempenho fluido.
É a integração entre conceitos de 2D, 3D e pós-processamento, reunidos em um único código.

Descrição das Funcionalidades:
1. Convolução 2D
- Renderiza a imagem em quadrado 2D;
- Cinco (seis com o filtro neutro) Kernels implementados no Fragment Shader;
- Shaders: Normal (Neutro), Blur, Sharpen, Bordas, Emboss e Outline;
- Alternância de filtros via teclado (teclas de 1 à 6);
- Tonalidade cinza: clique esquerdo;
- Reset: clique direito;
- Todo o processamento acontece na GPU.

2. Manipulação 3D
- Câmera com: posição, yam/pitch, velocidade;
- Movimentação: W, A, S, D;
- Rotação: setas;
- Subir/Descer: shift e espaço;
- Renderização com matriz MVP;
- Textura carregada com Pillow.

3. Aplicação conjunta
- Integração;
- Convolução 2D dentro do Shader 3D;
- Movimento completo da câmera;
- Sistema de filtros;
- FPSCounter oficial;
- V-Sync ativado.

4. FPSCounter (Extra)
- Utiliza a classe oficial passada em aula;
- Médias móveis;
- Estatísticas (min, max, media);
- Renderização na tela;
- Integração com V-Sync habilitado.

5. V-Sync (Extra)
- Ativado via glfw.swap;
- Elimina tearing;
- Estabiliza FPS;
- Reduz consumo de GPU;
- Mantém sincronia com a frequência do monitor.

Como rodar:
*Os requirements estão em um arquivo .txt no repositório*
- Arquivos necessários: conv2dcam3d.py, fpscounter.py, templo.png, requirements.txt
- Execução:
pip install -r requirements.txt
python conv2dcam3d.py
- Controles:
Trocar Kernel: teclas de 1 à 6
Tons de cinza: clique esquerdo
Reset: clique direito
Mover câmera frente\trás: teclas W e S
Mover câmera esquerda/direita: teclas A e D
Subir e descer a câmera: shift e espaço
Rotação: setas para os 4 lados

ROADMAP (Melhorias):
- Adicionar novos kernels (Gaussian Blur, Motion Blur, Boom, por exemplo)
- Adicionar menu para melhor navegação
- Carregamento dinâmico de imagens
- Múltiplos objetos/programas
- Melhorar a forma de reset da imagem
