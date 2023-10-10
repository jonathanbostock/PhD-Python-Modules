# JB's buffer calculator
def get_acid_conc(buffer_list, pH):
    total = 1000*10**(-pH)
   
    for conc, pKa in buffer_list:
       
        total += conc/(1+10**(pH-pKa))
   
    return total

def solve_buffer_system(buffer_list, acid_conc):

    error = 10
    working_pH = 8

    while abs(error) > 0.001:

        working_acid_conc = get_acid_conc(buffer_list, working_pH)
        #Error is the amount of acid that should be on the buffer at this pH
        #Minus the amount of acid we've added (with a term to include the ionization of water)
        error = working_acid_conc - (acid_conc - 1000*10**(-working_pH)+1000*10**(working_pH-14))
        working_pH += 0.1*error

    return working_pH
