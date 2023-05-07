clear all; clc; close all; 

a=[1,1.5;1,0.25];
theta=[0.001;1.8];

beta=50; 
tau=0.1; 
sigma_e=1; 

y_guess = [0,1,3; 0,0,1];
options = optimset('Display', 'off');
for i=1:3
y_1 = fsolve(@(y)wilson_cowan(y, a, theta, beta, tau), y_guess(:,i), options);
fixed_points(:,i) = y_1;
end 

disp(fixed_points);


u01=fixed_points(1,1) 
v01=fixed_points(2,1)


u03=fixed_points(1,3) 
v03=fixed_points(2,3)



%% figure 5
% clear all; clc; close all; 

filename=strcat('SSN_wilson_cowan_fig8');




tau=1; 
sigmaj=[0.1,0.4,1.1];
sigmaj=[0.35];

for j=1:3
clear y0

sigma=sigmaj(j);


% for j=1:length(tauj)
% clear y0

% tau=tauj(j); 

% a=[1.5,1;10,1];
a=[1,1.5;1,0.25];
theta=[0.001;1.8];
beta=50; 
sigma_e=1;
tau_e=1; 


tau_i=tau_e*tau; 
% tau_i=1; 
% tau_e=tau_i/tau; 
% tau_i=1;
tau_=[tau_e;tau_i];

sigma_i=sigma_e*sigma; 
sigma_=[sigma_e;sigma_i];

N=512; 

x=linspace(-25,25,N)'; 
dx=x(2)-x(1); 
T=100; 

tspan = [0 T]; 
dt=0.01; 

% noise=abs(randn(N,2));
% y0=repmat([u03,v03],N,1);%+0.1*noise; 
y0=zeros(N,2);
% y0(:,1)=0; %2
% y0(:,2)=0;
% y0(:,1)=0;
% y0(:,2)=0;
y0(450:N,1)=u03;
y0(450:N,2)=v03;
% y0(508:N,1)=3;
% y0(508:N,2)=15;


% y0=zeros(N,2);
% y0(500:N,1)=u03;
% y0(500:N,2)=v03;
% y0(500:N,1)=0.4;
% y0(500:N,2)=0;



% y0=repmat([u0,v0],N,1);
% NN=round(N/2); 
% index=randi([1,N],1,NN);
% y0(index,1)=u01;
% y0(index,2)=v01;

options = odeset('MaxStep',dt,'RelTol', 1e-6, 'AbsTol', 1e-6);
[t,y] = ode45(@(t,y)wilson_cowan_1d(x, y, a, theta, beta, sigma_, tau_), tspan, y0,options);

y=reshape(y,[],N,2);
u = y(:,:,1);
v = y(:,:,2); 

tmin=25;
tmax=30; 
[~,N0]=min(abs(t-tmin));
[~,N1]=min(abs(t-tmax));
t1=t(N0:N1); 

% % figure 3b
figure('Position',[0,600,400,300])
A=u';
imagesc(A);
colormap(hot);
colorbar;
txt={strcat("\tau = ",num2str(tau)),strcat("\sigma = ",num2str(sigma))}
text(500,50,txt,'FontSize',16,'Color','w')
% xlim([N0,N1])
% ylim([0,512])
set(gca, 'YDir', 'reverse'); % Reverse the y-axis direction
xlabel('Time');
ylabel('Population Number');
xticks(linspace(1, length(t), 5));
xticklabels(string(linspace(0, t(end), 5)));
yticks(linspace(0, 500, 6)); % Set y-ticks at 5 equally spaced points
set(gca,'FontSize',16,'LineWidth',2) 
fig_name=strcat(filename,'_B_','tau=',num2str(tau),'_sigma=',num2str(sigma),'.png');
saveas(gcf,fig_name)
close all; 

i=460;
ui=u(:,i);
vi=v(:,i);



%%%%%%%%%%%%%%%%%%%

% compute the v value for the u nullcline
u_all = linspace(-1,13,100);
% u_all=0;
for i=1:length(u_all)
    u=u_all(i);
    v_(i,:)=fzero(@(v)u_nullcline(u,v,a,theta,beta),0); 
end 

% compute the u value for the v nullcline
% v_all = logspace(-2,5,500); 
% v_all=linspace(-2,15,500);
% for i=1:length(v_all)
%     v=v_all(i);
%     u_(i,:)=fzero(@(u)v_nullcline(u,v,a,theta,beta),2); 
% end 


u_ = linspace(-1,13,100);
for i=1:length(u_)
    u=u_(i);
    v_all(i,:)=fzero(@(v)v_nullcline(u,v,a,theta,beta),0.01); 
end 


figure('Position',[0,600,400,300])
h(1)=plot(u_all,v_,'r-','LineWidth',2)%,'DisplayName','u-nullcline');
hold on; 
h(2)=plot(u_,v_all,'b-','LineWidth',2)%,'DisplayName','v-nullcline');
hold on; 
plot(ui,vi,'LineWidth',2)
txt={strcat("\tau = ",num2str(tau)),strcat("\sigma = ",num2str(sigma))}
text(0.1,10.5,txt,'FontSize',16)
xlabel('u')
ylabel('v')
xlim([-0.1,13])
ylim([-1,12])
% legend('location','best')
set(gca,'FontSize',16,'LineWidth',2) 
fig_name=strcat(filename,'_C_','tau=',num2str(tau),'_sigma=',num2str(sigma),'.png');
saveas(gcf,fig_name)
% fig_name=strcat(filename,'_C',num2str(j));
% saveas(gcf,fig_name,'png')
close all; 


end 



%%

function D = Deter(omega, u0, v0, a, beta, sigma)
    Ke=Kern_ft(omega,sigma(1));
    Ki=Kern_ft(omega,sigma(2));
    D = 1+beta*(a(2,2)*Ki*v0*(1-v0)-Ke*u0*(1-u0))+...
        beta^2*(a(1,2)-a(2,2))*u0*(1-u0)*v0*(1-v0)*Ki.*Ke;
end 

function Kernw = Kern_ft(omega,sigma)
    Kernw = 1/2/sigma*2/sigma./(1/sigma^2+omega.^2);
end 

% eq (2) in the paper 
function dydt = wilson_cowan(y, a, theta, beta, tau)
    % u = y(1), v = y(2); 
    dydt = zeros(2,1);
    I1 = a(1,1)*y(1) - a(1,2)*y(2) - theta(1); 
    I2 = a(2,1)*y(1) - a(2,2)*y(2) - theta(2); 
    dydt(1) = -y(1) + Fun(I1,beta); 
    dydt(2) = (-y(2) + Fun(I2,beta))/tau; 
end 

% eq (1) in the paper 
function dydt = wilson_cowan_1d(x, y, a, theta, beta, sigma, tau)
    % u = y(:,1), v = y(:,2);
%     dydt = zeros(length(x),2);
    N=length(x); 
    dx=x(2)-x(1); 
    y=reshape(y,[],2);
    Ke=Kern(x,sigma(1));
    Ki=Kern(x,sigma(2));

%     conv_e=conv(Ke,y(:,1), 'same')*dx;
%     conv_i=conv(Ki,y(:,2), 'same')*dx;
%     conv_e=conv(Ke,y(:,1), 'valid')*dx;
%     conv_i=conv(Ki,y(:,2), 'valid')*dx;

%     % periodic boundary condition 
%     conv_e = cconv(Ke, y(:, 1), N)*dx;
%     conv_i = cconv(Ki, y(:, 2), N)*dx;

%     reflecting boundary condition 
    ext_y = [flipud(y(1:N-1,:)); y; flipud(y(end-N+1:end,:))];
    conv_e = conv(Ke, ext_y(:, 1), 'same') * dx;

    if sigma(2)==0
        conv_i=y(:,2); 
    else 
    conv_i = conv(Ki, ext_y(:, 2), 'same') * dx;
    end 

    I1 = a(1,1)*conv_e - a(1,2)*conv_i - theta(1); 
    I2 = a(2,1)*conv_e - a(2,2)*conv_i - theta(2); 
    dydt(:,1) = (-y(:,1) + Fun(I1,beta))/tau(1); 
    dydt(:,2) = (-y(:,2) + Fun(I2,beta))/tau(2); 
    dydt=dydt(:);
end 


function f = u_nullcline(u, v, a, theta, beta)
    I1 = a(1,1)*u - a(1,2)*v - theta(1); 
    f = - u + Fun(I1,beta);
end

function f = v_nullcline(u, v, a, theta, beta)
    I2 = a(2,1)*u - a(2,2)*v - theta(2); 
    f = - v + Fun(I2,beta);
end


function F = Fun(I,beta)
%     F = 1./(1+exp(-beta*I)); 
    F = 0.8*(max(I,0)).^3; 
end 

function K = Kern(x,sigma)
    K = 1/2/sigma*exp(-abs(x)./sigma); 
end



function F=F_fun(z,a,theta)
    P=1/a(1,1)*(det(a)*max(z,0).^3+a(2,2)*(z+theta(1))-a(1,1)*theta(2));
    F=a(1,1)*max(z,0).^3-a(1,2)*max(P,0).^3-z-theta(1);
end 


