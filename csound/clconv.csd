<CsoundSynthesizer>
<CsOptions>
--opcode-lib=./libclconv.dylib
</CsOptions>
<CsInstruments>

ksmps = 512
0dbfs = 1

gift ftgen 0, 0, 0, 1, "pianoc2.wav", 0, 0, 1
;tablew 1, 0, gift

instr 1
ipsize = 512
idev =  1; /* device number */
ain mpulse 0.5, 1
ain1 diskin "fox.wav", 1, 0, 1
asig clconv ain1/60, gift, 1, idev
;asig dconv ain1/60, ftlen(gift), gift
  out(asig)

endin

instr 2
ipsize = 512
idev =  1; /* device number */
;ain1 diskin "fox.wav", 1, 0, 1
ain1 mpulse 0.5, 1
ain2 diskin "beats.wav", 1, 0, 1
asig cltvconv ain1, ain2, 1, 1, 1, sr, 1
  out(asig)

endin


</CsInstruments>
<CsScore>
i1 0 10
</CsScore>
</CsoundSynthesizer>

