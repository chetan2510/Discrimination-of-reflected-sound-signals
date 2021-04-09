Kc = [-100:0.01:100]
% y1 = abs(-0.864576 + 0.0676*Kc.^2 + 0.19136*Kc)
% y2 = abs(0.09568 -0.588256*Kc + 0.864576)
% plot(Kc,y1)
% hold on
% plot(Kc,y2)
% legend

y3 = abs(1-Kc/4);
y4 = abs(-Kc+(Kc.^2)/2);

plot(Kc,y3)
hold on
plot(Kc,y4)
legend()

