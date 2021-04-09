numD = 70*[1 2], denD = [1 10]
numG = 1       , denG = [1 1  0]
sys1 = tf(numD,denD)*tf(numG,denG)
sysCL= feedback(sys1,1)
step(sysCL)