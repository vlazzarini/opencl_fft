<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libclconv.dylib
</CsOptions>
<CsInstruments>

ksmps = 64
0dbfs = 1

gift ftgen 0, 0, 1024, 1, "pianoc2.wav", 0, 0, 1
;gift ftgen 0, 0, 256, 2, 1
;tablew 1, 0, gift

instr 1
ipsize = 256
idev =  1; /* device number */
ain mpulse 0.5, 1
ain1 diskin "fox.wav", 1, 0, 1
;asig clconv ain1, gift, ipsize, idev
asig ftconv ain1, gift, ipsize
  out(asig/60)

endin

instr 2
ipsize = 8192
idev =  1; /* device number */
ain1 diskin "fox.wav", 1, 0, 1
;ain2 oscili 0.5, 440
;ain1 mpulse 0.5, -1024
ain2 diskin "beats.wav", 1, 0, 1
asig cltvconv ain1, ain2, 1, 1, 8192, filelen("beats.wav")*sr, idev
;asig tvconv ain1, ain2, 1, 1, ipsize, ipsize*4
aout = asig/100
  out(aout)

fout "clconv.aif", 6, aout

endin


</CsInstruments>
<CsScore>
i2 0 360
</CsScore>
</CsoundSynthesizer>

