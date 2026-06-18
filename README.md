# aes_enem

## Labels

species:

- Alstonia Scholaris
- Citrus limon
- Jatropha curcas
- Mangifera indica
- Ocimum basilicum
- Platanus orientalis
- Pongamia Pinnata
- Psidium guajava
- Punica granatum
- Syzygium cumini
- Terminalia Arjuna

margin:

- entire
- indented
- lobed
- serrate
- serrulate
- undulate

shape:

- elliptical
- lanceolate
- oblong
- obovate
- ovate

texture:

- glossy
- leathery
- smooth
- rough

## Rules

if margin == 'serrate': return 'Ocimum basilicum'               
elif margin == 'indented': return 'Jatropha curcas'             
elif margin == 'lobed': return 'Platanus orientalis'            
elif margin == 'serrulate': return "Citrus limon"               
elif margin == 'entire':
    if shape == 'ovate': return 'Pongamia Pinnata'              
    elif shape == 'lanceolate': return 'Mangifera indica'       
    elif shape == 'oblong': return 'Syzygium cumini'            
    elif shape == 'obovate': return "Psidium guajava"           
    else:
        if texture == 'leathery': return "Alstonia Scholaris"   
        elif texture == 'rough': return "Terminalia Arjuna"     
        elif texture == 'glossy': return "Citrus limon"         
        else: return "Punica granatum"                          
else:
    if shape == 'elliptical': return 'Terminalia Arjuna'        
    elif shape == 'lanceolate': return "Mangifera indica"       
    else: return 'Syzygium cumini'                              

## Space

0 -> 1   : (0,0,1)        : 

1 -> 21  : (4,,) (0,0,0)  : (4,0,0) (4,0,1) (4,0,2) (4,0,3)
                          : (4,1,0) (4,1,1) (4,1,2) (4,1,3)
                          : (4,2,0) (4,2,1) (4,2,2) (4,2,3)
                          : (4,3,0) (4,3,1) (4,3,2) (4,3,3)
                          : (4,4,0) (4,4,1) (4,4,2) (4,4,3) (0,0,0) 

2 -> 20  : (1,,)          : (1,0,0) (1,0,1) (1,0,2) (1,0,3)
                          : (1,1,0) (1,1,1) (1,1,2) (1,1,3)
                          : (1,2,0) (1,2,1) (1,2,2) (1,2,3)
                          : (1,3,0) (1,3,1) (1,3,2) (1,3,3)
                          : (1,4,0) (1,4,1) (1,4,2) (1,4,3)

3 -> 8 : (0,1,) (5,1,)    : (0,1,0) (0,1,1) (0,1,2) (0,1,3)
                          : (5,1,0) (5,1,1) (5,1,2) (5,1,3)

4 -> 20  : (3,,)          : (3,0,0) (3,0,1) (3,0,2) (3,0,3)
                          : (3,1,0) (3,1,1) (3,1,2) (3,1,3)
                          : (3,2,0) (3,2,1) (3,2,2) (3,2,3)
                          : (3,3,0) (3,3,1) (3,3,2) (3,3,3)
                          : (3,4,0) (3,4,1) (3,4,2) (3,4,3)

5 -> 20  : (2,,)          : (2,0,0) (2,0,1) (2,0,2) (2,0,3)
                          : (2,1,0) (2,1,1) (2,1,2) (2,1,3)
                          : (2,2,0) (2,2,1) (2,2,2) (2,2,3)
                          : (2,3,0) (2,3,1) (2,3,2) (2,3,3)
                          : (2,4,0) (2,4,1) (2,4,2) (2,4,3)

6 -> 4  : (0,4,)          : (0,4,0) (0,4,1) (0,4,2) (0,4,3)
 
7 -> 4  : (0,3,)          : (0,3,0) (0,3,1) (0,3,2) (0,3,3)

8 -> 1  : (0,0,2)         :
 
9 -> 24 : (0,2,) (5,,)    : (0,2,0) (0,2,1) (0,2,2) (0,2,3)
                          : (5,0,0) (5,0,1) (5,0,2) (5,0,3)
                          : (5,1,0) (5,1,1) (5,1,2) (5,1,3)
                          : (5,2,0) (5,2,1) (5,2,2) (5,2,3)
                          : (5,3,0) (5,3,1) (5,3,2) (5,3,3)
                          : (5,4,0) (5,4,1) (5,4,2) (5,4,3)
      
10 -> 5 : (0,0,3) (5,0,)  : (5,0,0) (5,0,1) (5,0,2) (5,0,3) (0,0,3)