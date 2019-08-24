<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libclconv.dylib
</CsOptions>
<CsInstruments>

ksmps = 1
0dbfs = 1

gift ftgen 0, 0, 0, 1, "pianoc2.wav", 0,0,1
;tablew 1, 0, gift

instr 1
ipsize = 512
idev =  1; /* device number */
ain mpulse 0.5, 7
;ain diskin "fox.wav", 1, 0, 1
asig clconv ain, gift, ipsize, idev
  out(asig)

endin

instr 2
ipsize = 512
idev =  1; /* device number */
ain1 diskin "fox.wav", 1, 0, 1
ain2 diskin "beats.wav", 1, 0, 1
asig cltvconv ain1, ain2/200, 1, 1, ipsize, filelen("beats.wav")*sr, 1
  out(asig)

endin


</CsInstruments>
<CsScore>
i2 0 20
</CsScore>
</CsoundSynthesizer>

