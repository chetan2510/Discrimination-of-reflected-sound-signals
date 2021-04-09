% defining Kc
Kc = 1;

nomD = 1      ; denD = [1 0 0];
nomG = Kc*[1 0.2]; denG = [1 2];

%function
sysOP         = tf(nomD,denD)*tf(nomG,denG);
sysBare       = tf(nomD,denD)*tf(nomG,denG)/Kc;
sysCL         = feedback(sysOP,1);
sysCL_digital_e = c2d(sysCL,1,'foh');
sysCL_digital_t = c2d(sysCL,1,'tustin');

%root locus
figure('Name', 'Root Locus of the system')
rlocus(sysBare)
grid on
grid minor
%plot step response
figure('Name', 'Transfer function response')
step(sysCL)
hold on
step(sysCL_digital_e)
hold on
step(sysCL_digital_t)
legend;
grid on;
grid minor