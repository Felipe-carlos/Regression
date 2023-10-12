# Título: Regressão linear usando o descenso do gradiente
# Autor: Felipe C. dos Santos
#
# Descrição: faz uma regressão linear em um modelo linear de uma variavel para aproximar a função f_true usando o descenso do gradiente
print('Felipe Carlos dos Santos')

alfa = float #taxa de aprendizado
epocas = 5000 #numero de epocas
n = 1 #grau polinomial do modelo


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


plt.rc('axes', grid=True)

def f_true(x):
    return 2 + 0.8 * x

# conjunto de dados {(x,y)}
xs = np.linspace ( -3 , 3 , 100)
ys = np.array ( [f_true( x ) + np.random.randn() * 0.5 for x in xs] )


#hipotese:
def hip(thetas,xs,n):  #recebe uma lista com theta0 e theta1 onde h(x)= theta0 + theta1 * x in xs e n o grau do modelo
  resultado= []
  for x in xs:
      y=0
      for _ in range(n+1):              #função generalizada para regressão polinomial
          y += thetas[_]* (x **_)
      resultado.append(y)
  return resultado

def cost(thetas,xs,ys): #recebe uma lista com theta0 e theta1 e os valores xs e ys do conjunto de dados e retorna o valor do erro quadratico médio entre h(x) e y
    return (1/(2*len(xs)))*(sum((hip(thetas,xs,n)-ys))**2)

def gradiente_step(thetas,xs,ys,alfa,n):   #recebe os thetas, xs, ys e alfa e retorna os novos valores dos thetas
    new_theta=[]
    for pos,theta in enumerate(thetas):
        new_theta.append(theta - alfa * (1/len(xs) * sum((hip(thetas,xs,n)-ys)*xs**pos)))  #função generalizada para regressão polinomial

    return new_theta

def print_modelo(thetas,xs,ys): #recebeo numero da fugura n,lista com theta0 e theta1 e os dados xs e ys e plota o grafico dos dados reais, modelo e dados com ruído
    plt.figure()
    plt.plot(xs,[f_true(x) for x in xs],label='Função real')
    plt.plot(xs,ys,label='Dados com ruído')
    plt.plot(xs,hip(thetas,xs,n),label='Modelo')
    plt.title(f'Taxa de aprendizagem = {alfa}')
    plt.legend()
    #plt.savefig(f'n={n}_Taxa de aprendizagem = {alfa}.png')

def print_3d(): #apenas printa um grafico 3d dos valores theta0, theta1 e custo
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    theta0 = np.linspace(-5,5,200)
    theta1 = theta0
    Theta0, Theta1 = np.meshgrid(theta0,theta1)

    # Plot the surface.
    for t0 in Theta0:
        b = [((t0 + t1) -2.8)**2 for t1 in Theta1]


    surf = ax.plot_surface(Theta0, Theta1,np.array(b), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('Custo')


    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

# -------- execução principal do programa:

for  alfa in [0.9, 0.1, 0.0001]:
    thetas = np.zeros(n+1) #chute inicial dos valres de theta em 0

    historico_thetas = [thetas]      #guarda o valor de theta0 atraves das epocas
    custo = [cost(thetas,xs, ys)]       #guarda o valor do custo atraves das epocas

    for x in range(epocas):
        thetas = gradiente_step(thetas,xs,ys,alfa,n)
        historico_thetas.append(thetas)
        custo.append(cost(thetas,xs,ys))
    print_modelo(thetas,xs,ys)

#-----------------------------Outros Plots:---------------------------
    #-------------------------Função de custo para Taxa de aprendizagem
    plt.figure()
    plt.plot(np.arange(epocas+1),custo)
    plt.title(f'Função de custo para Taxa de aprendizagem = {alfa}')
    plt.xlabel('Épocas')
    plt.ylabel('Custo')
    #plt.savefig(f'n={n}_Função de custo para Taxa de aprendizagem = {alfa}.png')
    # -------------------------Função de custo vs thetas
    """
    for run in range(n+1):
        x_axis=[linha[run] for linha in historico_thetas]
        plt.figure()
        plt.plot(x_axis,custo)
        plt.xlabel(f'Theta_{run}')
        plt.ylabel('Custo')
        plt.title(f'Função de custo vs Theta{run} para alfa = {alfa}')
        plt.savefig(f'n={n}_Função de custo vs Theta{run} para alfa = {alfa}.png')
        
   """
    # -------------------------Thetas vs Epocas para alfa
    plt.figure()
    for run in range(n + 1):
        y_axis = [linha[run] for linha in historico_thetas]

        plt.plot(np.arange(epocas+1), y_axis,label=f'Theta_{run}')
        plt.xlabel('Épocas')
        plt.ylabel(f'Theta_{run}')
        plt.title(f'Thetas vs Epocas para alfa = {alfa}')
        plt.legend()
    #plt.savefig(f'n={n}_Thetas vs Epocas para alfa = {alfa}.png')

plt.show()