%% REINFORCEMENT LEARNING FOR TRIPLE-WELL MODEL

clearvars
%% Set parameters for RL:
    Nepoch=2;    % number of epochs
    eps=linspace(1,0,Nepoch);   %epsilon-greedy 
    Nstep=20;   % number of update steps in each epoch              
    Tmin=0.2;Tmax=1.3;  % temperature range   
    T_int=0.1; %temperature mesh size
    s_int=0.1; %s mesh size
    A=[-0.05 0 0.05]; %actions on temperature
    s_tar=0.91 ;  %s value of target
    alpha=0.7;  gamma=0.9; %learning rate and discount factor

%% Set parameters for the triple-well:
    N_BD=1000; %update steps at fixed T 
    dt=1;%fixed, vary tau
    tau=500;  
    noises=0.022; %standard deviation for noise of s 
    noiseT=0.00;  %standard deviation for noise of T 
    pd=makedist('Normal','mu',0,'sigma',noises);
    pdT=makedist('Normal','mu',0,'sigma',noiseT);
    muG=[0.16 0.55 0.88]; sigmaG=[0.2 0.11 0.11]; %mean and std of the Gaussian distribution
    irng=randi(1000); rng(irng);
    
%% Save the input:
    inputfile = fopen('00input.dat','wt');
    fprintf(inputfile,'%-10s\t  %u\n','epoch',Nepoch);
    fprintf(inputfile,'%-10s\t','epsilon');    fprintf(inputfile,'%.3f\t',eps);
    fprintf(inputfile,'\n%-10s\t  %u\n','nstep',Nstep);
    fprintf(inputfile,'%-1-0s\t  %.3f\n','Tmin', Tmin);
    fprintf(inputfile,'%-10s\t  %.3f\n','Tmax', Tmax);
    fprintf(inputfile,'%-10s\t  %.3f\n','s_interval', s_int);
    fprintf(inputfile,'%-10s\t  %.3f\n','T_interval', T_int); 
    fprintf(inputfile,'%-10s\t', 'A'); fprintf(inputfile,'%.3f\t',A);
    fprintf(inputfile,'\n%-10s\t  %.3f\n','s_target',s_tar);
    fprintf(inputfile,'%-10s\t  %.3f\n','alpha',alpha);
    fprintf(inputfile,'%-10s\t  %.3f\n','gamma',gamma);    
    fprintf(inputfile,'%-10s\t  %u\n','rng',irng);    fclose(inputfile);

    
%% Training process: 
Ss=(s_int/2):s_int:1;            
ST=(Tmin+T_int/2):T_int:Tmax;
Q=zeros(length(Ss),length(ST),length(A));   
for iepoch=1:Nepoch 
    iepoch
    ieps=eps(iepoch); 
    traindata=zeros(Nstep+1,3); %data is set each episode
    
    s1=rand;  T1=round(Tmin+(Tmax-Tmin)*rand,2);
    s=s1; T=T1;
    
    xx1=abs(Ss-s1); xx2=abs(ST-T1);
    iss1=find(xx1==min(xx1)) ;    iss2=find(xx2==min(xx2)) ;
    iss1=iss1(randi(length(iss1)));
    iss2=iss2(randi(length(iss2))); 
    istep=0; 
    traindata(istep+1,:)=[istep T1 s1];
            %[istep T state state_target action reward]
for istep=1:Nstep
    eps_rand=rand; 
    if eps_rand >= ieps %choose action at maxQ    
        ia=find(Q(iss1,iss2,:)==max(Q(iss1,iss2,:)));
        if length(ia) ~= 1  
            ia=ia(randi(length(ia)));
        end               
        T1=T+A(ia); %stay there if out of range of T
        if T1 < Tmin-0.001;     T1=T; 
        elseif T1 > Tmax+0.001; T1=T; 
        end
        T1=round(T1,2);        
       
    else %else random action
        ia=randi(length(A)); T1=T+A(ia); 
        if T1 < Tmin-0.001;     T1=T; 
        elseif T1 > Tmax+0.001; T1=T; 
        end
        T1=round(T1,2);       
    end
    
    for j=1:N_BD %update state after "N_BD" steps
        [s1, ]=fstate_3well(s,T1,dt,tau,noises,muG,sigmaG);
        s=s1;
    end

    % calculate the reward:   
    rwd_1=-(s1-s_tar)^2;
        
    % update Q table by s_t, a_t, R_(t+1), s_(t+1)       
    xx1=abs(Ss-s1); xx2=abs(ST-T1);
    iss1_1=find(xx1==min(xx1)) ;    iss2_1=find(xx2==min(xx2)) ; 
    iss1_1=iss1_1(randi(length(iss1_1)));
    iss2_1=iss2_1(randi(length(iss2_1)));
    Q(iss1,iss2,ia)=  Q(iss1,iss2,ia)...
                + alpha*(rwd_1 + gamma*max(Q(iss1_1,iss2_1,:))-Q(iss1,iss2,ia)) ;
    traindata(istep+1,:)=[istep T1 s1];
      
    
    iss1=iss1_1;     iss2=iss2_1; T=T1;
end

%% save data each epoch
save(strcat('traindata_epoch',num2str(iepoch,'%u'),'.dat'),'traindata','-ascii')
reshapeQ=reshape(Q,[],1);
save(strcat('train_Q_epoch',num2str(iepoch,'%u'),'.dat'),'reshapeQ','-ascii')

%% plot data s and T each epoch
figure; 
title(strcat('Training data, epoch',num2str(iepoch),', eps=',num2str(ieps,'%.2f')))
subplot(2,1,1);
plot(traindata(:,1),traindata(:,2),'.-k','LineWidth',1)
ylabel 'T'; ylim([Tmin-0.1 Tmax+0.1]); ytickformat('%.2f')
set(gca,'FontSize',14) ; 
subplot(2,1,2);
plot(traindata(:,1),traindata(:,3),'.-m','LineWidth',1)
ylabel 'State s'; ylim([Tmin-0.1 Tmax+0.1]) ;ytickformat('%.2f')
set(gca,'FontSize',14)
xlabel 'Update step'; 
savefig(strcat('fig_epoch',num2str(iepoch,'%u'),'_sT.fig')); close

%% plot the policy each epoch
col(1,:)=[0 0 1]; col(3,:)=[1 0 0];col(2,:)=[0.5 0.5 0.5];
figure; hold on
for iSp=1:length(Ss)
for iSv=1:length(ST) 
        ia=find(Q(iSp,iSv,:)==max(Q(iSp,iSv,:)));
    if length(ia)==1
        scatter(Ss(iSp),ST(iSv),550,col(ia,:),'s','filled','MarkerEdgeColor',[0 0 0])
    end

end
end
hold off; axis equal; box on;  grid on;
xlim([0 1]);ylim([0 1.5]); %box on
xlabel 's'; ylabel 'T';
title(strcat('Policy after epoch',num2str(iepoch)))
savefig(strcat('fig_epoch',num2str(iepoch,'%u'),'_policy.fig')); close

end


function [sn1, fvalue]=fstate_3well(s,T, dt,tau,noise,muG,sigmaG)
pd=makedist('Normal','mu',0,'sigma',noise);

pd1=-pdf(makedist('Normal','mu',muG(1),'sigma',sigmaG(1)),s)*T^2;
pd2=-pdf(makedist('Normal','mu',muG(2),'sigma',sigmaG(2)),s)*(1.5-T)^4*1;
pd3=-pdf(makedist('Normal','mu',muG(3),'sigma',sigmaG(3)),s)*(1.4-T)^4*1.8;

fvalue=(pd1+pd2+pd3)/tau;
df=-pd1.*(s-muG(1))/sigmaG(1)^2 -pd2.*(s-muG(2))/sigmaG(2)^2-pd3.*(s-muG(3))/sigmaG(3)^2;
sn1=s-df*dt/tau + random(pd);
if sn1>1; sn1=1;
elseif sn1<-0 ; sn1=0;
end

end