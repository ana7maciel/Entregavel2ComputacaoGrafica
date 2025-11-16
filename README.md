# Entregavel2ComputacaoGrafica

Disciplina: Computação Gráfica | Alunos: Ana Karolina Maciel, Carlo e Eduardo Fagundes

Link Repositório:

Elevator Pitch:
Esta aplicação demonstra, de forma interativa e em tempo real, o uso de filtros de convolução aplicados a uma imagem de um templo. O usuário pode alternar dinamicamente entre cinco kernels diferentes, ativar o modo em tons de cinza e restaurar a imagem original com apenas um clique.
Além disso, o projeto inclui uma câmera 3D completa, com movimentação em primeira pessoa e rotação do ponto de vista, proporcionando uma experiência imersiva. Para complementar, foram implementados um contador de FPS e V-Sync, garantindo desempenho fluido.
É a integração entre conceitos de 2D, 3D e pós-processamento, reunidos em um único código.

Descrição das Funcionalidades:
1. Convolução 2D
- Renderiza imagem em quadrado 2D;
- Cinco (seis com o filtro neutro) implementados no Fragment Shader;
- Shaders: Normal (neutro, Blur, Sharpen, Bordas, Emboss, Outline;
- Alternância de filtros via teclado (teclas de 1 a 6);
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

ROADMAP (Melhorias):
- Adicionar novos kernels (Gaussian Blur, Motion Blur, Boom, por exemplo)
- Adicionar menu para melhor navegação
- Carregamento dinâmico de imagens
- Múltiplos objetos/programas
- Melhorar a forma de reset da imagem
