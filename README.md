# Object Conter -open_CV
 create object counter using open_cv


- **Reference**

- [Documentation][def]
- [Github Main repo][def2]

Hi All,
To solve this issue, I just replace that line with these two lines of code :
cnts = cv2.findContours(FrameThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

please add import imutils as well

regards



[def]: icrosoftazuresponsorships.com/Balance
[def2]: https://github.com/phfbertoleti/ContadorObjetosEmMovimento/tree/master