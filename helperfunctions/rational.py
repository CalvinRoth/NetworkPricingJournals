class RationalF:
    def __init__(self, topCoeff, bottomCoeff):
        self.top = topCoeff.copy()
        self.bottom = bottomCoeff.copy() 

    def eval(self, x):
        t = 0
        b = 0 
        for i,c in enumerate(self.top):
            t += c * (x ** i)
        for j,c  in enumerate(self.bottom):
            b += c * (x ** j)
        return t/b
    
    def poles(self):
        if(len(self.bottom) > 3):
            print("Not supported, returning 0")
            return 0 
        if(len(self.bottom) == 1): ## Bottom is a constant 
            return 0
        elif(len(self.bottom) == 2):
            return - self.bottom[0] / self.bottom[1]
        else:
            c = self.bottom[0]
            b = self.bottom[1]
            a = self.bottom[2]
            discrim = (b*b) - (4 * a * c)
            discrim = discrim **(1/2)
            sol1 = (-b) + discrim
            sol2 = (-b) - discrim
            sol1 = sol1 / (2*a)
            sol2 = sol2 / (2*a)
            return sol1, sol2
    def __repr__(self):
        t = ""
        b = ""
        for i,c in enumerate(self.top):
            t =  t +  str(c) + "x^" + str(i) + " "
        t = t[0:-2]
        for j,c in enumerate(self.bottom):
            b += b + str(c) + "x^" + str(j) + " "
        t = t[0:-2]
        return t + "/" + b  
