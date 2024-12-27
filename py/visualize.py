# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 14:47:50 2021

@author: Dell
"""

from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np

clist = ['r','g','b','purple','orange','pink','cyan','grey','black']
mlist = ['o', 'v', '^', '<', '>', '1', '2', '3', '4']

def show(title=None, xlabel=None, ylabel=None):
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()
    
def drawSegment(ax, p1, p2, grade=100):
    g = np.linspace(0, 1, grade)
    x = p1[0] + g * (p2[0] - p1[0])
    y = p1[1] + g * (p2[1] - p1[1])
    ax.plot(x, y, c='gray', linewidth=0.1, alpha=1)

def scatter2D(X, y, marker='.', title=None):
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    if type(y) is int:
        label2color = clist[y]
    elif type(y) is str:
        label2color = y
    else:
        label2color = [clist[y_i] for y_i in y]
    ax.scatter(X[:, 0], X[:, 1], c=label2color, marker=marker, s=1)
    # if title:
    #     plt.title(title)
    plt.show()
    fig.savefig('result/' + title + '.pdf', format='pdf')
    
    
def scatterByInd2D_graph(X, F, title='ScatterInd', marker='.'):
    F = np.array(F)
    red = np.array([1, 0, 0, 1])
    green = np.array([0, 1, 1, 1])
    
    def drawCol(ax, cx, cy, h):
        r = 0.1
        x = np.linspace(-r, r, 5)
        z = np.linspace(0, h, 5)
        X, Z = np.meshgrid(x, z)
        Y = np.sqrt(r**2 - X**2)
        ax.plot_surface(X+cx, Y+cy, Z, color = (h+1)/2 * red + (-h+1)/2 * green)
    
    label = F[:,0] - np.sum(F, axis=1).reshape(-1)/2
    label /= np.amax(abs(label))
    
    fig = plt.subplots(figsize=(6,6), dpi=300)
    ax = plt.axes(projection='3d')
    ax.view_init(elev=30, azim=-25)
    for i in range(X.shape[0]):
        drawCol(ax, X[i,0], X[i,1], float(label[i]))
    # if title:
    #     plt.title(title)
    
    x = np.linspace(np.amin(X[:,0]), np.amax(X[:,0]), 10)
    y = np.linspace(np.amin(X[:,1]), np.amax(X[:,1]), 10)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, Z=X*0, color='grey', alpha=0.2) 

    plt.show()
    # fig.savefig('result/' + title + '.pdf', format='pdf')
    
    
def scatterByLabels2D_graph(X, y, l, title='Scatter', marker=('.', 'x')):
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    clist = ['r','g','b','#8B0000', '#008B00', '#00008B']
    label2color = [clist[y_i] for y_i in y[l:]]
    ax.scatter(X[l:, 0], X[l:, 1], c=label2color, marker=marker[0], s=1)
    label2color = [clist[y_i+3] for y_i in y[:l]]
    ax.scatter(X[:l, 0], X[:l, 1], c=label2color, marker=marker[1])
    # if title:
    #     plt.title(title)
    plt.show()
    fig.savefig('result/' + title + '.pdf', format='pdf')
    
def scatterKNN_graph(X, y, l, knn, title='KNN', marker=('.', 'x')):
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    clist = ['r','g','b','#8B0000', '#006400', '#00008B', 'pink','cyan','purple','orange','grey','black']
    
    for i in range(knn.shape[0]):
        for j in knn[i]:
            drawSegment(ax, X[i], X[j])
            
    label2color = [clist[y_i] for y_i in y[l:]]
    ax.scatter(X[l:, 0], X[l:, 1], c=label2color, marker=marker[0], s=1)
    label2color = [clist[y_i+3] for y_i in y[:l]]
    ax.scatter(X[:l, 0], X[:l, 1], c=label2color, marker=marker[1])
    # if title:
    #     plt.title(title)
    plt.show()
    fig.savefig('result/' + title + '_KNN.pdf', format='pdf')
    
def scatter2D_with_l(X, y, l, title='samples', marker=('x', '.')):
    scatter2D(X[:l], y[:l], marker=marker[0])
    scatter2D(X[l:], y[l:], marker=marker[1])

def multiple_line_with_legend(x, Y, title, legends="123456789"):
    
    fig = plt.figure(figsize=(8,6), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    
    # for i in [0,1,-1,2,3,4,5]:
    for i in range(len(Y)):
        ax.plot(x, [y*100 for y in Y[i]], label=legends[i], marker=mlist[i])
    
    ax.set_xlabel("Number of labeled samples per class", fontsize=18)
    ax.set_ylabel("Classification Accuracy(%)", fontsize=18)
    ax.set(
       # xlim=(0, 1500),
       ylim=(0, 100),
       xticks=(x),
       )
    
    ax.legend(bbox_to_anchor=(1,0), loc='lower right', fontsize=14)
    fig.savefig('result/' + title + '.pdf', format='pdf')
    
def single_line(x, y):
    lines = []
    lines.append(plt.plot(x, y, c='r', marker='o'))
    
    # plt.xticks(x)

# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# Y = [[0.7478427452940196, 0.7919693435521493, 0.8157390434927511, 0.8812333333333333, 0.9063843973995664, 0.9070190063354453, 0.9154243788560947, 0.9463475650433623, 0.9422351959966638, 0.9375875875875875, 0.9377023193726014, 0.9383177570093458, 0.9351861125020864, 0.931677796327212, 0.934747036233094, 0.9391950567802271, 0.9390679806246867, 0.9401520213832274, 0.9426232247284879, 0.9466076203208557], [0.7442945194069632, 0.7801774408530491, 0.8297887852024664, 0.8721770833333333, 0.8851725287547925, 0.9066105368456153, 0.9165666166416541, 0.919221147431621, 0.9350375312760635, 0.9269686353019686, 0.9268980477223426, 0.9236481975967958, 0.9225838758137206, 0.9221035058430717, 0.9231925196193019, 0.9299014696058784, 0.9324703524302655, 0.9341797527564317, 0.944937343358396, 0.9470254010695187], [0.5162418790604698, 0.601107964011996, 0.6476170638226962, 0.730525, 0.7622103683947323, 0.8040263421140381, 0.8131732532933131, 0.8265510340226818, 0.853277731442869, 0.8647814481147813, 0.8860587351910562, 0.883820093457944, 0.8843598731430479, 0.8861769616026711, 0.8867256637168142, 0.8942050768203073, 0.902330048438283, 0.9036501837621115, 0.9275187969924813, 0.9252172459893048], [0.8189655172413793, 0.8501166277907364, 0.8694134310948176, 0.900925, 0.9251375229204868, 0.9363787929309769, 0.9522594630648656, 0.9562541694462976, 0.9617931609674729, 0.9640056723390057, 0.9640830969464375, 0.9684579439252335, 0.9710983141378733, 0.9698247078464106, 0.9716229754550009, 0.9710921843687375, 0.974185735760815, 0.9744570664884732, 0.9757226399331664, 0.9753676470588235]]
# leg=['LLGC', 'HF', 'GF(G)', 'GF(A)']
# multiple_line_with_legend(x, Y, 'Swarm_label', leg)
# show()

def twinLine2D_graph(x, y1, y2, xlabel=None, y1label=None, y2label=None, title='title'):
    xlabel = xlabel or 'Number of anchors'
    y1label = y1label or 'Classification Accuarcy(%)'
    y2label = y2label or 'Computing Time(second)'
    
    fig = plt.figure(figsize=(8,6), dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)
    
    ax1.plot(x, y1, color = 'blue', label = y1label)
    
    ax1.set_xlabel(xlabel, fontsize=18)
    ax1.set_ylabel(y1label, fontsize=18)
    ax1.set(# xlim=(0, 2000),
       ylim=(0, 100),
       # xticks=[100,500,1000,1500],
       )
    
    ax2 = ax1.twinx()
    
    ax2.plot(x, y2, color = 'red', label = y2label)
    
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel(y2label, fontsize=18)
    ax2.set(# xlim=(0, 2000),
       ylim=(0, np.amax(y2)*1.1),
       # xticks=np.arange(100, 2001, 100),
       )

    fig.legend(loc='lower right', bbox_to_anchor=(1,0), bbox_transform=ax1.transAxes, fontsize=14)
    # plt.title()
    # plt.show()
    fig.savefig('result/' + title + '.pdf', format='pdf')

# x=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595]
# y1=[63.37815126050421, 70.23529411764706, 71.74789915966386, 71.27731092436974, 70.52100840336134, 71.3781512605042, 69.8655462184874, 68.45378151260503, 68.18487394957982, 68.20168067226892, 68.6890756302521, 68.63865546218489, 68.15126050420169, 67.52941176470588, 67.64705882352942, 67.32773109243698, 67.39495798319327, 66.85714285714285, 67.10924369747899, 66.85714285714285, 66.4873949579832, 66.1344537815126, 66.15126050420169, 66.15126050420169, 65.89915966386555, 65.96638655462185, 65.84873949579831, 65.89915966386555, 65.68067226890756, 65.46218487394958, 65.61344537815125, 65.5798319327731, 65.5126050420168, 65.32773109243696, 65.32773109243696, 65.31092436974791, 65.26050420168067, 65.34453781512605, 65.21008403361344, 65.21008403361344, 65.24369747899159, 65.15966386554621, 65.12605042016807, 64.99159663865545, 65.02521008403362, 64.99159663865545, 64.89075630252101, 64.90756302521008, 64.84033613445378, 64.77310924369748, 64.72268907563026, 64.57142857142857, 64.57142857142857, 64.6890756302521, 64.52100840336135, 64.48739495798318, 64.57142857142857, 64.43697478991596, 64.3529411764706, 64.45378151260505, 64.48739495798318, 64.40336134453781, 64.33613445378151, 64.28571428571429, 64.40336134453781, 64.47058823529413, 64.4201680672269, 64.36974789915966, 64.38655462184875, 64.15126050420167, 64.18487394957984, 64.0672268907563, 64.08403361344538, 64.15126050420167, 64.13445378151262, 64.15126050420167, 64.01680672268908, 64.05042016806723, 64.01680672268908, 64.05042016806723, 63.7983193277311, 64.13445378151262, 63.94957983193278, 63.8655462184874, 63.781512605042025, 63.81512605042017, 63.781512605042025, 63.94957983193278, 63.983193277310924, 64.08403361344538, 64.13445378151262, 63.966386554621856, 63.966386554621856, 63.91596638655462, 63.91596638655462, 63.88235294117648, 63.9327731092437, 63.81512605042017, 63.69747899159665, 63.630252100840345, 63.66386554621849, 63.69747899159665, 63.81512605042017, 63.84873949579833, 63.9327731092437, 63.9327731092437, 64.01680672268908, 64.05042016806723, 64.0, 63.9327731092437, 63.831932773109244, 63.76470588235294, 63.54621848739497, 63.49579831932773, 63.46218487394959, 63.46218487394959, 63.411764705882355, 63.411764705882355, 63.27731092436976]
# y2=[0.0021963432312011715, 0.0021353588104248045, 0.002191758155822754, 0.0024935007095336914, 0.0018947124481201172, 0.0019945144653320313, 0.002292299270629883, 0.0020990610122680665, 0.001997518539428711, 0.0020942449569702148, 0.002294588088989258, 0.0022980928421020507, 0.002393770217895508, 0.0019945144653320313, 0.002295851707458496, 0.002003049850463867, 0.0020944356918334963, 0.0021012544631958006, 0.002491402626037598, 0.0022988319396972656, 0.0022940635681152344, 0.0021770954132080077, 0.0024201393127441405, 0.0025044918060302735, 0.0023990631103515624, 0.0024973154067993164, 0.00259251594543457, 0.0025948286056518555, 0.0025936126708984374, 0.0026964187622070313, 0.0026929616928100587, 0.002860355377197266, 0.0027914047241210938, 0.0026938199996948244, 0.0027933120727539062, 0.0025921106338500977, 0.0029921770095825196, 0.003090190887451172, 0.003094029426574707, 0.0031917810440063475, 0.002992081642150879, 0.003192901611328125, 0.0028941154479980467, 0.0033974647521972656, 0.00309605598449707, 0.003581118583679199, 0.003497314453125, 0.0034618616104125977, 0.003491973876953125, 0.003291463851928711, 0.003890705108642578, 0.004213452339172363, 0.004088640213012695, 0.004189133644104004, 0.004488396644592285, 0.004488134384155273, 0.004395103454589844, 0.004495763778686523, 0.004693436622619629, 0.004193663597106934, 0.004687881469726563, 0.0050795555114746095, 0.005292367935180664, 0.005292510986328125, 0.0053899288177490234, 0.005585217475891113, 0.005446887016296387, 0.006286263465881348, 0.005983805656433106, 0.006377410888671875, 0.005589890480041504, 0.0065851926803588865, 0.006440281867980957, 0.00638267993927002, 0.00668337345123291, 0.006869077682495117, 0.006850504875183105, 0.006681466102600097, 0.007085680961608887, 0.007389283180236817, 0.0070811033248901365, 0.006972956657409668, 0.007741618156433106, 0.007380127906799316, 0.008588600158691406, 0.007582354545593262, 0.007987022399902344, 0.00818023681640625, 0.007991981506347657, 0.008680582046508789, 0.008382105827331543, 0.008935546875, 0.009080290794372559, 0.00877680778503418, 0.008779788017272949, 0.00963895320892334, 0.008771562576293945, 0.009772014617919923, 0.009733152389526368, 0.010622525215148925, 0.00997779369354248, 0.010571980476379394, 0.00936877727508545, 0.011463403701782227, 0.010266423225402832, 0.010968780517578125, 0.010557866096496582, 0.012974023818969727, 0.010381793975830078, 0.01186830997467041, 0.011456727981567383, 0.01226966381072998, 0.011450815200805663, 0.012265992164611817, 0.011968255043029785, 0.013268542289733887, 0.012082314491271973, 0.013463640213012695, 0.012141013145446777]

# twinLine2D_graph(x, y1, y2, title='Balance_anchor')
# show()
     
# from v2
def geo3D_graph(data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    xx = np.linspace(0, len(data), len(data))
    yy = np.linspace(0, len(data), len(data))
    
    X,Y = np.meshgrid(xx, yy)
    ax.plot_surface(X, Y, data)
    
    ax.set(xlabel='beta',
       ylabel='alpha',
       zlabel='accuracy',
       # xlim=(0, 9),
       # ylim=(0, 9),
       # zlim=(0, 1)
       # zlim=(0.85, 0.9),
       # xticks=np.arange(0, 10, 2),
       # yticks=np.arange(0, 10, 1),
       # zticks=np.arange(0, 10, 1)
       )
    ax.view_init(30, -120)
    
    # plt.title('bar3D')
    plt.show()

def bar3D_graph(data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    dx = dy = 0.38
    def cube(x,y,h):
        xx = np.linspace(x-dx, x+dx, 2)
        yy = np.linspace(y-dy, y+dy, 2)
        zz = np.linspace(0, h, 2)
        
        X, Y = np.meshgrid(xx, yy)
        ax.plot_surface(X, Y, X*0+h, color='cyan', alpha=1)
        
        X, Z = np.meshgrid(xx, zz)
        ax.plot_surface(X, X*0+y-dy, Z, color='mediumturquoise', alpha=1)
        
        Y, Z = np.meshgrid(yy, zz)
        ax.plot_surface(Y*0+x-dx, Y, Z, color='lightseagreen', alpha=1)
        
    for j in range(len(data)):
        for i in range(len(data[0])):
            cube(i, j, data[i,j])
    
    ax.set(xlabel='beta',
       ylabel='alpha',
       zlabel='accuracy',
       # xlim=(0, 9),
       # ylim=(0, 9),
       zlim=(0, 1)
       # zlim=(0.85, 0.9),
       # xticks=np.arange(0, 10, 2),
       # yticks=np.arange(0, 10, 1),
       # zticks=np.arange(0, 10, 1)
       )
    ax.view_init(30, -120)
    
    # plt.title('bar3D')
    plt.show()
    
def errorBar2D_graph(data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(data[:,0], data[:,1], color = 'cyan')
    ax.errorbar(data[:,0], data[:,1], data[:,(2,3)].T, fmt = '.k', ecolor = 'lightgrey', elinewidth = 3, capsize = 0)
    
    plt.xscale('log')
    ax.set(xlabel='number of labeled samples',
       ylabel='accuracy',
       # xlim=(0, 70000),
        ylim=(0, 1)
       # xticks=np.arange(0, 10, 2),
       # yticks=np.arange(0, 10, 1),
       )
    
    
def scatter2D_graph(X, y, title='samples', marker='.'):
    clist = ['r','g','b','purple','cyan','orange','pink','grey','black']
    plt.title(title)
    label2color = [clist[y_i] for y_i in y]
    plt.scatter(X[:, 0], X[:, 1], c=label2color, marker=marker, s=1)
    plt.show()
    
    