function out_struct = res_to_struct (path,n)

s = ['a = load (''',path,''', ''-ascii'');'];
eval(s)
T = length(a) / n ;
a=reshape(a,T,n);
out_struct = struct ; 
for i = 1:n
    q = num2str(i);
   s= ['temp',q, '= find (a(:,', q , '));'] ;
   eval (s);   
   
end
for i = 1:n
    s = ['out_struct(i).Target = temp', num2str(i),';'];
    eval (s) 
    largest_s = [];
    l = ['maximum = max (['];
    for p = 1:n  
        if p~=i
           l = [l [' length(temp', num2str(p),  ')']];
           
        end
    end
     l = [l ']);'];
    eval (l)
    nn = 1 ; 
   for j = 1:n
       if j~=i
           js = num2str(j);
           nns = num2str(nn);
           s = ['difference = maximum - length(temp', js,');'];
           eval (s);
     s = ['temp(:,',nns,') = [temp', js,';zeros(difference,1)];'];
     eval (s);
     nn = nn + 1 ;
       end
   end
    out_struct(i).Ref = temp;
   clear temp ;
end
