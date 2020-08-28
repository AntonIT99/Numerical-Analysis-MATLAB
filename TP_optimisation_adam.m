clear;
clf; 
N=100;
p=10;

%%%%%%%%%%  fonctionnelle de Rosenbrock %%%%%%%%%%%%%
J = @(x) (x(1)-1).^2 + p*(x(1)^2 - x(2)).^2;
gradient_J = @(x) [2*(x(1)-1) + 2*p*(x(1)^2 - x(2))*2*x(1);-2*p*(x(1)^2 - x(2))]; 

%%%%%%%%%%%%%%%% initialisation %%%%%%%%%%%%%%%%%%%%%%%%%%
Xk = [0;0]; 
Yk = Xk;  

%%%%%%%%%% paramètre de la méthode du gradient
epsilon = 10^(-3);
N_max = 10000;
k=0; 
alpha = 0.0025;

%%%%initialisation + paramètre  méthode d'ADAM %%%%%%%
delta = 10^(-3); 
b1=0.9; 
b2 =0.95;

gk = gradient_J(Xk); 
mt = gk; 
vt = sum(gk.^2);

while  k<149
%     norm(gk) > 10^(-4) 
%     && k<N_max
k = k+1;

%%%%%%%%%%%% méthode du gradient %%%%%%%%%%%%%%%%%%%%%
gk = gradient_J(Xk);
Xk = Xk - alpha*gk;

%%%%%%%%%%%% Méthode d'ADAM %%%%%%%%%%%%%%%%%%%%%%%%%%
gk = gradient_J(Yk);
mt = b1*mt + (1-b1)*gk;
vt = b2*vt + (1-b2)*sum(gk.^2);
% vt = b2*vt + (1-b2)*((gk.^2));
Yk = Yk - alpha*sqrt(1 - b2)/(1-b1)*mt/(sqrt(vt)+ delta);
% Yk = Yk - alpha*( (1/(1-b1^k))*mt./(sqrt(vt/(1-b2^k))+ delta));

%%%%%%%%%%%% Calcul de J %%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
Erreur_gradient(k) = J(Xk);
Erreur_ADAM(k) = J(Yk);
end

disp(Yk);

%-- Affichage du log de l'erreur
figure(1);
plot(1:k,log(Erreur_gradient),'b');
hold on;
plot(1:k,log(Erreur_ADAM),'m');
legend('log(J(\theta)),  Gradient', 'log(J(\theta^k)), ADAM')






