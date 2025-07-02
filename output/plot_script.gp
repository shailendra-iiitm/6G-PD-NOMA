set terminal pngcairo enhanced font 'arial,10' fontscale 1.0 size 800, 600
set output 'oma_vs_noma_comparison.png'
set title 'OMA vs NOMA Performance Comparison (2-user scenario)'
set xlabel 'SNR (dB)'
set ylabel 'Throughput (bps/Hz)'
set grid
set key top left
plot 'oma_data.txt' with lines lw 2 title 'OMA (TDMA/FDMA)', \
     'noma_data.txt' with lines lw 2 title 'NOMA Sum Rate', \
     'noma_user1_data.txt' with lines lw 2 title 'NOMA User1 (Weak)', \
     'noma_user2_data.txt' with lines lw 2 title 'NOMA User2 (Strong)'
