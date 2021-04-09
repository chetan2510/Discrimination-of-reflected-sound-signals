Ts=1; 
pole = 0.42+0.45i;

chosenpole = 0.00005;
% gain = 60 damping = 0.509 freq = 0.954 rad/s

% from coefficient a
a = -1/chosenpole;

% choose a and calculate b to let "pole" appears in the root locus
otherphase = -pi - angle(pole+1) + 2*angle(pole-1) +angle(pole + 1/a)

% calculate coefficient b
b = tan(otherphase)/(0.45-0.42*tan(otherphase));

disp ("zero is: " + -1/b)
disp ("pole is: " + chosenpole)
disp ("a = " + a + " and b = " + b)

figure('Name', 'root locus')
sys     = tf((Ts^2/2)*[1 1],[1 -2 1],Ts);
control = tf([1 1/b],[1 1/a], Ts);
sysOP = sys*control
rlocus(sysOP)

figure('Name', 'step transfer function response')
Kc = 0.063 ;%plotting closed loop feedback
sysCL = feedback(0.477*sysOP,1);
step(sysCL)