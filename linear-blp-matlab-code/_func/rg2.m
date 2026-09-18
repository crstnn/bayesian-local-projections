function x = rg2(a)
    
b = a-1;
c = 3*a - 0.75;
accept = 0;
while accept == 0
u = rand;
v = rand;
w = u*(1-u);
y = sqrt(c/w)*(u-0.5);
x = b+y;
if x >= 0
z = 64*(w^3)*(v^2);
if z <= ( 1-(2*y^2)/x )
    accept = 1;
end    
if accept == 0
if log(z) <= 2*(b*log(x/b) - y)
    accept = 1;
end    
end
end
end