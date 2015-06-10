% 3D mesh plot 
figure(1); 

hold on;

pointA = [0, 0, 0];
pointB = [0, 100, 1];
pointC = [100, 0, 1];
points=[pointA' pointB' pointC']; 
fill3(points(1,:),points(2,:),points(3,:),'r');
grid on;
alpha(0.3);

pointA = [0, 0, 0];
pointB = [0, 100, 2];
pointC = [100, 0, 2];
points=[pointA' pointB' pointC']; 
fill3(points(1,:),points(2,:),points(3,:),'r');
alpha(0.3);

view([-20, 50]);
