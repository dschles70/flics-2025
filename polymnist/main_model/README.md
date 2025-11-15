# e2

Copy from `e1`, make p(y) in x-VAE depending on (d,z)

Start without synthetic examples \
(0) nz = 16, ny = 16 \
(1) nz = 16, ny = 64 \
(2) nz = 16, ny = 256 -> bad d ... otherwise the best so far

(3) as (2) but with learned iconst \
(4) ^, introduce "last operation"

bug ? found: iconst was learnable only in the encoder \
cancel (3)/(4)

make learnable iconst the member of x (common for encoder and decoder) \
make prior depending also on c \
keep "last op" tracking

(3) as said \
learned iconst does not help, even seems to make c a bery bit worse

___

cancel all, revert to constant iconst, start from the beginning \
keep lastop-filtering, x-prior depends on everything

(0)-(3) nz = 16, ny = 32/64/128/256 \
(4)-(7) nz = 32, ny = 32/64/128/256 \
-> nz = 32 makes d even wose, others seemingly the same


(0): \
0.98099995, 0.98939997,  0.987,  0.98819995,  0.98679996 -> 0.98648


(3): \
0.9801, 0.9906,  0.991,  0.98969996, 0.9905 -> 0.988379992

(4) as (3) with generator

(5) real data again, another (general) d-model \
(6) special d-model, d not depending on (c,z) -> seems to work \
(7) ^, with cs-generator

(6): 0.9848, 0.9909, 0.99039996,  0.99069995, 0.98939997, 0.98924 \
(7): 0.9823, 0.987, 0.98759997, 0.98749995, 0.9863, 0.98613995

**The acual pair is (6)/(7)** 

(8)/(9) continue (6)/(7) for further 500k iterations \
(10)/(11) ^, continue slower

--> (9) unstable, then crashes

(10): \
0.9863, 0.99109995, 0.9913, 0.9903, 0.98969996, 0.98973995

(11): \
0.9849,  0.9885,  0.9888,  0.98829997,  0.98899996, 0.98789996

**The final pair is (10)/(11)** 
