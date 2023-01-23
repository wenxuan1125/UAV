                # pose estimation
                imgpoints= np.array([(x, y), (x, y+h), (x+w, y+h), (x+w, y)], np.float32)
                objpoints = np.array([(0, 0, 0), (0, 40, 0), (160, 40, 0), (160, 0, 0)], np.float32)
                retval, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, intrinsic, distortion)

                # cv2.putText( img, text to show, coordinate, font, font size, color, line width, line type )
                cv2.putText(frame, str(tvec[2]), (x + w, y + h+10), cv2.FONT_HERSHEY_SIMPLEX,\
                0.4, (0, 255, 0), 1, cv2.LINE_AA)
