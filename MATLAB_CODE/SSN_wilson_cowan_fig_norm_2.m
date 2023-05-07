clear all; clc; close all; 

a=[1,1.5;1,0.25];

theta=[0.001;1.8];
% theta=[1,1];

beta=50; 
tau=0.1; 
sigma_e=1; 

y_guess = [0,1,3; 0,0,1];
options = optimset('Display', 'off');
for i=1:3
y_1 = fsolve(@(y)wilson_cowan(y, a, theta, beta, tau), y_guess(:,i), options);
fixed_points(:,i) = y_1;
end 

% disp(fixed_points);


u01=fixed_points(1,1);
v01=fixed_points(2,1);


u03=fixed_points(1,3);
v03=fixed_points(2,3);


fixed_points


%% figure 
% clear all; clc; close all; 

filename=strcat('SSN_wilson_cowan_fig_norm_2');


clear y0 ;




tau=0.1;
sigma=0; 


tau_e=1; 
tau_i=tau_e*tau; 
tau_=[tau_e;tau_i];

sigma_i=sigma_e*sigma; 
sigma_=[sigma_e;sigma_i];


a=[1,1.5;1,0.25];
% theta=[0.001;1.8];
beta=50; 


N=512; 

x=linspace(-25,25,N)'; 
dx=x(2)-x(1); 
tmax=30; 

tspan = [0 tmax]; 
dt=0.01; 

% noise=abs(randn(N,2));
% y0=repmat([u03,v03],N,1);%+0.1*noise; 
% y0(:,1)=0; %2
% y0(:,2)=0;
% y0(:,1)=0;
% y0(:,2)=0;
% % % % y0(450:N,1)=u03;
% % % % y0(450:N,2)=v03;
% y0(508:N,1)=3;
% y0(508:N,2)=15;

% y0=repmat([u0,v0],N,1);
% NN=round(N/2); 
% index=randi([1,N],1,NN);
% y0(index,1)=u01;
% y0(index,2)=v01;

c=2;  % the maximum for c is 14, the minimum is 1. 
x0=10; 
sigma_ff=2; 
sigma_ff=0.125*dx;
l=40; 
% h=G_fun(x0,x,sigma_ff)+1/c;
h=1./(1+exp(-(x+l/2)/sigma_ff)).*(1-1./(1+exp(-(x-l/2)/sigma_ff)));
% c=1;
% h=ones(size(x)); 

% options = odeset('MaxStep',dt,'RelTol', 1e-6, 'AbsTol', 1e-6);
% [t,y] = ode45(@(t,y)wilson_cowan_1d(x, y, a, theta, beta, sigma_, tau_), tspan, y0,options);

y0=zeros(N,2);
y0((256-30):(256+30),1)=u03;
y0((256-30):(256+30),2)=v03;

% y0(450:N,1)=u03;
% y0(450:N,2)=v03;



[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c, h, a, theta, beta, sigma_, tau_);


y=reshape(y_values,[],N,2);
u = y(:,:,1);
v = y(:,:,2); 


% % figure 3b
figure('Position',[0,600,900,300])
subplot(1,2,1)
plot(c*h,'LineWidth',2) 
xlim([0,N])
xlabel('Population Number');
ylabel('Input')
set(gca,'FontSize',16,'LineWidth',2) 
subplot(1,2,2)
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
fig_name=strcat(filename,'_1','.png');
saveas(gcf,fig_name)

%%
        figure; 
c_all=[1:2:10]; 
for j=1:length(c_all)
        % center
        c2=c_all(j); 
        h2=1./(1+exp(-(x+l0/2)/sigma_rf)).*(1-1./(1+exp(-(x-l0/2)/sigma_rf)));  

        % surround 
        c1=5; 
        h1=1./(1+exp(-(x+l/2)/sigma_rf)).*(1-1./(1+exp(-(x-l/2)/sigma_rf)));  
        c1h1=c1*(h1-h2); 


        c=1; 
        h=c1h1+c2*h2; 

        plot(c*h)
        hold on; 
end 

%% fig 3e left 
 
l0=5*dx;
%l_all=l0*[1,2,3,4,5,12,20];  % the size for the surround stimulus 
l_all=logspace(-2,log10(5),10);
c_all=[0,0.1,0.5,1,2,3:4:10]%,20,25]; 

l_all=logspace(0,log10(5),4);
l_all=[l0,logspace(-2,log10(2),4)];
l_all=l0*[1,2,3,4,5,12,20]; 
l_all=l0*[1,3,5,10,20,100];
% c_all=[1:5:10]*2; 
% l_all=[1:3:15];

clear re_all ri_all

for i=1:length(l_all)
%     figure;
    for j=1:length(c_all)

        [i,j]
        l=l_all(i); 
        sigma_rf=0.125*dx*2.5; 
%         sigma_rf=2;
        
        % center 
        c2=c_all(j); 
        h2=1./(1+exp(-(x+l0/2)/sigma_rf)).*(1-1./(1+exp(-(x-l0/2)/sigma_rf)));  

        % surround 
        c1=5; %5
        h1=1./(1+exp(-(x+l/2)/sigma_rf)).*(1-1./(1+exp(-(x-l/2)/sigma_rf)));
        c1h1=c1*(h1-h2); 


        c=1; 
        h=c1h1+c2*h2; 


%         plot(c1*h1)
%         hold on; 
%         plot(c*h)
       

        %[t,y_values,I] = SSN(dt, tmax, y0, c, h, W, tau, k, n);
        [t,y_values,I] = SSN_wc(x,dt, tmax, y0, c, h, a, theta, beta, sigma_, tau_);
        re = y_values(:,:,1);
        ri = y_values(:,:,2); 

        re_all(i,j)=re(end,256);
        ri_all(i,j)=ri(end,256);
    end 
end 

%%
figure('Position',[0,600,400,300])
plot(c_all,re_all,'*-')
xlabel('Center Stimulus Strength')
ylabel('Firing Rate')
legend('Center Alone','2x','3x','4x','5x','12x','20x','location','northwest')



%% fig 3e right
c=50; 
l0=1.9;
l_all=l0*[1,2,3,4,5,12,20];  % the size for the surround stimulus 
c_all=[0:10:50]; 

clear re_all ri_all

for i=1:length(l_all)
    for j=1:length(c_all)

        [i,j]
        l=l_all(i); 
        sigma_rf=0.125*dx; 

        
        % center 
        c2=c_all(j); 
        h2=1./(1+exp(-(x+l0/2)/sigma_rf)).*(1-1./(1+exp(-(x-l0/2)/sigma_rf)));  

        % surround 
        c1=50; 
        h1=1./(1+exp(-(x+l/2)/sigma_rf)).*(1-1./(1+exp(-(x-l/2)/sigma_rf)));
        c1h1=c1*(h1-h2); 


        c=1; 
        h=c1h1+c2*h2; 

        [t,y_values,I] = SSN(dt, tmax, y0, c, h, W, tau, k, n);
        re = y_values(:,:,1);
        ri = y_values(:,:,2); 

        re_all(i,j)=re(end,101);
        ri_all(i,j)=ri(end,101);
    end 
end 


figure('Position',[0,600,400,300])
plot(c_all,ri_all)
xlabel('Center Stimulus Strength')
ylabel('Firing Rate')
legend('Center Alone','2x','3x','4x','5x','12x','20x','location','northwest')








%% fig 1g simulation 

phi1=10; 
phi2=-10; 
sigma_ff=2;

c=10; 

% tau=[1,0.5];
% sigma_=[1;0.5];

y0=zeros(N,2);
n0=30;
y0((256-n0):(257+n0),1)=u03;
y0((256-n0):(257+n0),2)=v03;

unique(y0(257:end,:)-flip(y0(1:256,:)))

c1=c;
h1=G_fun(phi1,x,sigma_ff)+1/c1;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c1, h1, a, theta, beta, sigma_, tau_);
y=reshape(y_values,[],N,2);
re_1 = y_values(end,:,1);
ri_1 = y_values(end,:,2); 
figure; plot(re_1)

c2=c; 
h2=G_fun(phi2,x,sigma_ff)+1/c2;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c2, h2, a, theta, beta, sigma_, tau_);
re_2 = y_values(end,:,1);
ri_2 = y_values(end,:,2); 

c=1;
h=c1*h1+c2*h2;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c, h, a, theta, beta, sigma_, tau_);
re_ = y_values(end,:,1);
ri_ = y_values(end,:,2); 


w1=0.5; w2=0.5; 


xdata=x; 
ydata=re_; 
fun=@(x,xdata)(x(1)*re_1+x(2)*re_2); 
x0=[w1,w2];
lb = [];
ub = [];
options=optimoptions('lsqcurvefit','Display','off');
fitting_para_e=lsqcurvefit(fun,x0,xdata,ydata,lb,ub,options)
re_fitted=fun(fitting_para_e,xdata);


xdata=x; 
ydata=ri_; 
fun=@(x,xdata)(x(1)*ri_1+x(2)*ri_2); 
x0=[w1,w2];
lb = [];
ub = [];
options=optimoptions('lsqcurvefit','Display','off');
fitting_para_i=lsqcurvefit(fun,x0,xdata,ydata,lb,ub,options)
ri_fitted=fun(fitting_para_i,xdata);


% figure; 
% plot(c1*h1)
% hold on; 
% plot(c2*h2)
% hold on; 
% plot(c*h)


% fig 1g plot 
figure('Position',[0,600,800,400])
subplot(4,2,1)
plot(re_1,'LineWidth',2)
xlim([0,N])
ylim([0,14])
set(gca,'XTickLabel',[]);
set(gca,'FontSize',12,'LineWidth',1)

subplot(4,2,3)
plot(re_2,'LineWidth',2)
xlim([0,N])
ylim([0,14])
set(gca,'XTickLabel',[]);
set(gca,'FontSize',12,'LineWidth',1)

subplot(4,2,5)
plot(re_,'LineWidth',2)
hold on; 
ylim([0,14])
plot(re_fitted,'k--','LineWidth',2)
% txt={strcat("W1 = ",num2str(round(fitting_para_e(1)*100)/100)),strcat("W2 = ",num2str(round(fitting_para_e(2)*100)/100))};
txt={strcat("W = ",num2str(round(fitting_para_e(1)*100)/100))};
text(10,12,txt,'FontSize',10,'Color','k')
% ylim([6,8])
xlim([0,N])
ylim([0,14])
set(gca,'XTickLabel',[]);
set(gca,'FontSize',12,'LineWidth',1)

subplot(4,2,7)
plot(re_,'k','LineWidth',3)
hold on; 
plot(re_1+re_2,'b','LineWidth',1)
hold on; 
plot((re_1+re_2)/2,'r','LineWidth',1)
ylim([0,14])
xlim([0,N])
xlabel('Preferred Orientation')
xlabel('Population Number');
set(gca,'FontSize',12,'LineWidth',1)


subplot(4,2,2)
plot(ri_1,'LineWidth',2)
xlim([0,N])
ylim([0,7])
set(gca,'XTickLabel',[]);
set(gca,'FontSize',12,'LineWidth',1)


subplot(4,2,4)
plot(ri_2,'LineWidth',2)
xlim([0,N])
ylim([0,7])
set(gca,'XTickLabel',[]);
set(gca,'FontSize',12,'LineWidth',1)


subplot(4,2,6)
plot(ri_,'LineWidth',2)
hold on; 
plot(ri_fitted,'k--','LineWidth',2)
% txt={strcat("W = ",num2str(fitting_para_i(1)))};
% txt={strcat("W1 = ",num2str(round(fitting_para_i(1)*100)/100)),strcat("W2 = ",num2str(round(fitting_para_i(2)*100)/100))};
txt={strcat("W = ",num2str(round(fitting_para_i(1)*100)/100))};
text(420,6,txt,'FontSize',10,'Color','k')
ylim([0,7])
xlim([0,N])
set(gca,'XTickLabel',[]);
set(gca,'FontSize',12,'LineWidth',1)


subplot(4,2,8)
plot(ri_,'k','LineWidth',3)
hold on; 
plot(ri_1+ri_2,'b','LineWidth',1)
hold on; 
plot((ri_1+ri_2)/2,'r','LineWidth',1)
xlim([0,N])
ylim([0,7])
% legend('Results','Mean','Sum')
xlabel('Preferred Orientation')
xlabel('Population Number');
set(gca,'FontSize',12,'LineWidth',1)
fig_name=strcat(filename,'_1g','.png');
saveas(gcf,fig_name)


%% fig 1h left

figure('Position',[0,600,600,400])

c_all=[10,10;12.5,7.5;15,5;19,1]; 

for i=1:4 

c1=c_all(i,1);
c2=c_all(i,2); 

phi1=-10;
h1=G_fun(phi1,x,sigma_ff)+1/c1;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c1, h1, a, theta, beta, sigma_, tau_);
re_1 = y_values(end,:,1);
ri_1 = y_values(end,:,2); 

phi2=10;
h2=G_fun(phi2,x,sigma_ff)+1/c2;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c2, h2, a, theta, beta, sigma_, tau_);
re_2 = y_values(end,:,1);
ri_2 = y_values(end,:,2); 

c=1; 
h=c1*h1+c2*h2-1;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c, h, a, theta, beta, sigma_, tau_);
re_ = y_values(end,:,1);
ri_ = y_values(end,:,2); 


subplot(2,2,i)
plot(re_1,'LineWidth',1)
hold on; 
plot(re_2,'LineWidth',1)
plot(re_,'k-','LineWidth',2)
ylim([0,18])
xlim([0,N])
txt={strcat("Input Ratio = ",num2str(c1),':',num2str(c2))};
text(230,16,txt,'FontSize',10,'Color','k')
set(gca,'FontSize',12,'LineWidth',1)
xlabel('Population Number');
ylabel('Excitatory Firing Rate')

end 
fig_name=strcat(filename,'_1h_left','.png');
saveas(gcf,fig_name)



%% fig 1h right 
clear w_all 
c_ratio=logspace(0,-2.5,10);

for i=1:length(c_ratio)

c1=20/(1+c_ratio(i));
c2=20-c1; 


h1=G_fun(phi1,x,sigma_ff)+1/c1;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c1, h1, a, theta, beta, sigma_, tau_);
re_1 = y_values(end,:,1);
ri_1 = y_values(end,:,2); 


h2=G_fun(phi2,x,sigma_ff)+1/c2;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c2, h2, a, theta, beta, sigma_, tau_);
re_2 = y_values(end,:,1);
ri_2 = y_values(end,:,2); 

c=1; 
h=c1*h1+c2*h2-1;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c, h, a, theta, beta, sigma_, tau_);
re_ = y_values(end,:,1);
ri_ = y_values(end,:,2); 


w1=0.5; w2=0.5; 


xdata=x; 
ydata=re_; 
fun=@(x,xdata)(x(1)*re_1+x(2)*re_2); 
x0=[w1,w2];
lb = [];
ub = [];
options=optimoptions('lsqcurvefit','Display','off');
fitting_para_e=lsqcurvefit(fun,x0,xdata,ydata,lb,ub,options);
re_fitted=fun(fitting_para_e,xdata);


xdata=x; 
ydata=ri_; 
fun=@(x,xdata)(x(1)*ri_1+x(2)*ri_2); 
x0=[w1,w2];
lb = [];
ub = [];
options=optimoptions('lsqcurvefit','Display','off');
fitting_para_i=lsqcurvefit(fun,x0,xdata,ydata,lb,ub,options);
ri_fitted=fun(fitting_para_i,xdata);


w_all(i,:)=fitting_para_e;
end 


figure('Position',[0,600,400,300])
plot(log10(c_ratio),w_all,'LineWidth',2)
set(gca, 'XDir', 'reverse'); % Reverse the y-axis direction
xlabel('Log_{10}((Stimulus 2)/(Stimulus 1))')
ylabel('Weights')
legend('Weight 1','Weight 2')
set(gca,'FontSize',16,'LineWidth',2) 
fig_name=strcat(filename,'_1h_right','.png');
saveas(gcf,fig_name)



%% fig 1i
clear w_all
% c_all=[[0.1:1:5],[10:5:50]];
c_all=[0.001,0.01,0.1,1:5:30];
clear w1_all w2_all 

for i=1:length(c_all)

c1=c_all(i);
c2=c1; 


h1=G_fun(phi1,x,sigma_ff)+1/c1;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c1, h1, a, theta, beta, sigma_, tau_);
re_1 = y_values(end,:,1);
ri_1 = y_values(end,:,2); 


h2=G_fun(phi2,x,sigma_ff)+1/c2;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c2, h2, a, theta, beta, sigma_, tau_);
re_2 = y_values(end,:,1);
ri_2 = y_values(end,:,2); 

c=1; 
h=c1*h1+c2*h2-1;
[t,y_values,I] = SSN_wc(x,dt, tmax, y0, c, h, a, theta, beta, sigma_, tau_);
re_ = y_values(end,:,1);
ri_ = y_values(end,:,2); 


w1=0.5; w2=0.5; 


xdata=x; 
ydata=re_; 
fun=@(x,xdata)(x(1)*re_1+x(2)*re_2); 
x0=[w1,w2];
lb = [];
ub = [];
options=optimoptions('lsqcurvefit','Display','off');
fitting_para_e=lsqcurvefit(fun,x0,xdata,ydata,lb,ub,options);
re_fitted=fun(fitting_para_e,xdata);


xdata=x; 
ydata=ri_; 
fun=@(x,xdata)(x(1)*ri_1+x(2)*ri_2); 
x0=[w1,w2];
lb = [];
ub = [];
options=optimoptions('lsqcurvefit','Display','off');
fitting_para_i=lsqcurvefit(fun,x0,xdata,ydata,lb,ub,options);
ri_fitted=fun(fitting_para_i,xdata);


w1_all(i,:)=[fitting_para_e(1),fitting_para_i(1)];
w2_all(i,:)=[fitting_para_e(2),fitting_para_i(2)];

end 



figure('Position',[0,600,400,300])
plot(c_all,w1_all,'LineWidth',2)
hold on; 
% plot(c_all,w2_all,'LineWidth',2)
xlabel('Stimulus strength')
ylabel('Weights')
legend('E cells','I cells')
set(gca,'FontSize',16,'LineWidth',2) 


%% fig 1j simulation 
c_all=[10:10:50];
sigma_ff_all=[[0.1:0.5:5],[5:5:100]]; 
phi=45; 
clear re_all
for i=1:length(c_all)
    for j=1:length(sigma_ff_all)
        c=c_all(i); 
        sigma_ff=sigma_ff_all(j);
        h=G_fun(phi,x,sigma_ff);
        [t,y_values,I] = SSN_wc(x,dt, tmax, y0, c, h, a, theta, beta, sigma_, tau_);
        re = y_values(end,:,1);
        ri = y_values(end,:,2); 
        re_all(i,j)=re(phi);
    end 
end 

%% fig 1j plot
figure('Position',[0,600,600,300])
plot(sigma_ff_all,re_all./max(re_all,[],2),'LineWidth',2)
xlabel('Input Width')
ylabel('Firing Rate')
legend('10','20','30','40','50')







%%
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



% end 

%%
%function [t,y_values,I] = SSN_wc(x,dt, tmax, y0, c, h, a, theta, beta, sigma_, tau_)

function [t,y_values,I] = SSN_wc(x, dt, tmax, y0, c, h, a, theta, beta, sigma, tau)

%     re = y(:,1), ri = y(:,2);

    t=[dt:dt:tmax]'; 

    dx=x(2)-x(1); 

    N=length(y0);
    Nt=length(t);
    y_values=zeros(Nt,N,2);  
    y_values(1,:,:)=y0; 

    dydt = zeros(N,2);

    Iext=c*h;

    Ke=Kern(x,sigma(1));
    Ki=Kern(x,sigma(2));

    for i=1:Nt
    
        y=reshape(y_values(i,:,:),[],2);


    %     reflecting boundary condition 
        ext_y = [flipud(y(1:N-1,:)); y; flipud(y(end-N+1:end,:))];
        conv_e = conv(Ke, ext_y(:, 1), 'same') * dx;
    
        if sigma(2)==0
            conv_i=y(:,2); 
        else 
        conv_i = conv(Ki, ext_y(:, 2), 'same') * dx;
        end 
    
% G=G_fun(x,x',sigma_ori);
% W.Wee=Jee*G; 
% W.Wei=Jei*G;
% W.Wie=Jie*G;
% W.Wii=Jii*G;

%         conv_e = 

        Iee=a(1,1)*conv_e;
        Iei=a(1,2)*conv_i;
        Iie=a(2,1)*conv_e;
        Iii=a(2,2)*conv_i;

        I1=Iee-Iei-theta(1)*Iext; 
        I2=Iie-Iii-theta(2)*Iext; 

        dydt(:,1) = (-y(:,1) + Fun(I1,beta))/tau(1); 
        dydt(:,2) = (-y(:,2) + Fun(I2,beta))/tau(2); 
    
        y_values(i+1,:,:) = y_values(i,:,:)+dt*reshape(dydt,1,N,2);
    end 

        I.Ie=I1; 
        I.Ii=I2; 
        I.Iee=Iee; 
        I.Iei=Iei; 
        I.Iie=Iie; 
        I.Iii=Iii;         
        I.Iext=Iext; 
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


function G = G_fun(x1,x2,sigma)
    x=x1-x2;
    X=abs(x); 
%     X=min(abs(x),180-abs(x));
    G = 1/sigma/sqrt(2*pi)*exp(-X.^2/(2*sigma^2));
end 


function F=F_fun(z,a,theta)
    P=1/a(1,1)*(det(a)*max(z,0).^3+a(2,2)*(z+theta(1))-a(1,1)*theta(2));
    F=a(1,1)*max(z,0).^3-a(1,2)*max(P,0).^3-z-theta(1);
end 


