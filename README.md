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

if margin == 'serrate': return 'Ocimum basilicum'               Ocimum basilicum: margin serrate, shape any, texture any
elif margin == 'indented': return 'Jatropha curcas'             Jatropha curcas: margin indented, shape any, texture any
elif margin == 'lobed': return 'Platanus orientalis'            Platanus orientalis: margin lobed, shape any, texture any
elif margin == 'serrulate': return "Citrus limon"               Citrus limon: margin serrulate, shape any, texture any
elif margin == 'entire':
    if shape == 'ovate': return 'Pongamia Pinnata'              Pongamia Pinnata: margin entire, shape ovate, texture any
    elif shape == 'lanceolate': return 'Mangifera indica'       Mangifera indica: margin entire, shape lanceolate, texture any
    elif shape == 'oblong': return 'Syzygium cumini'            Syzygium cumini: margin entire, shape oblong, texture any
    elif shape == 'obovate': return "Psidium guajava"           Psidium guajava: margin entire, shape obovate, texture any
    else:
        if texture == 'leathery': return "Alstonia Scholaris"   Alstonia Scholaris: margin entire, shape elliptical, texture leathery   
        elif texture == 'rough': return "Terminalia Arjuna"     Terminalia Arjuna: margin entire, shape elliptical, texture rough   
        elif texture == 'glossy': return "Citrus limon"         Citrus limon: margin entire, shape elliptical, texture glossy   
        else: return "Punica granatum"                          Punica granatum: margin entire, shape elliptical, texture smooth    
else:
    if shape == 'elliptical': return 'Terminalia Arjuna'        Terminalia Arjuna: margin undulate, shape elliptical, texture any
    elif shape == 'lanceolate': return "Mangifera indica"       Mangifera indica: margin undulate, shape lanceolate, texture any
    else: return 'Syzygium cumini'                              Syzygium cumini: margin undulate, shape any, texture any

