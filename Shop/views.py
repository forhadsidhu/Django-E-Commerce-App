from django.shortcuts import render,redirect


import cv2
import numpy as np
import threading
from django.http import StreamingHttpResponse
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .forms import CreateUserform
from .models import Counter
from django.db.models import F


# authentication r jnno
from django.contrib.auth import  authenticate,login,logout
from django.contrib.auth import authenticate as auth_main, login as dj_login

#this is needed for making login compulsary and redirect to login
from django.contrib.auth.decorators import login_required
from django.views.decorators.gzip import gzip_page
from django.conf import settings
from django.http import JsonResponse
from .forms import Rev
import os
import cv2 as cv
from PIL import  Image
# Create your views here.




BASE_DIR = settings.BASE_DIR

"""
  name   sharif
  Database password sharifullah

"""


cascPath =  "haarcascade_frontalface_default.xml"





class VideoCamera():
    def __init__(self):
        self.video = cv.VideoCapture(0)

        '''
         One path
        
        '''
        img = Image.open('E:/Ecommerce/Django/static/img/download.jpg')
        img = img.convert("RGBA")

        pixdata = img.load()
        width, height = img.size
        for y in range(height):
            for x in range(width):
                if pixdata[x, y] == (255, 255, 255, 255):
                    pixdata[x, y] = (255, 255, 255, 0)

        img.save("img2.png", "PNG")

        '''
        Another path
        '''

        self.mask = Image.open('img2.png')
    def __del__(self):
        self.video.release()

    def face_eyes(self):


        self.face_cc = cv.CascadeClassifier(cv.data.haarcascades + cascPath)
        success, image = self.video.read()


        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


        faces = self.face_cc.detectMultiScale(gray, 1.3, 5)
        font = cv.FONT_HERSHEY_SIMPLEX

        background = Image.fromarray(image)


        for (x, y, w, h) in faces:
            resized_mask =self.mask.resize((w, h), Image.ANTIALIAS)
            # define offset for mask
            offset = (x, y)
            # pask mask on background
            background.paste(resized_mask, offset, mask=resized_mask)

            # cv.rectangle(image, (x, y), (x + w, y + h), (180, 255, 10), 2)
            # cv.putText(image, 'Face detected!', (x + w // 6, y - 15),
            #            font, 0.003 * w, (255, 180, 10), 2, cv.LINE_AA)
        image = np.asarray(background)
        ret, jpeg = cv.imencode('.jpg', image)
        return jpeg.tobytes()



def first_view(request):
    return render(request,'camera/first_view.html')




def gen(camera):
    i=0
    while True:
        frame = camera.face_eyes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        i+=1
        if i==100:
            # release camera and redirect to home page
            camera.__del__()
            break




@gzip_page
def face(request):
    try:
        return render(request, 'play.html')

        #return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")
    except:

        print("yes face detectionn fail camer......................")
        return  redirect('home')



# redirect first to login page
@login_required(login_url='log')
def home(request):
    return render(request,'home/home.html')


"""Creating face dataset

"""
def create_dataset(username):
    #print request.POST
    #userId = request.POST['userId']
    print (cv2.__version__)
    # Detect face
    #Creating a cascade image classifier
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
    #camture images from the webcam and process and detect the face
    # takes video capture id, for webcam most of the time its 0.
    cam = cv2.VideoCapture(0)

    # Our identifier
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is
    # id = userId
    # Our dataset naming counter
    sampleNum = 0
    id=username
    # Capturing the faces one by one and detect the faces and showing it on the window
    while(True):
        # Capturing the image
        #cam.read will return the status variable and the captured colored image
        ret, img = cam.read()
        #the returned img is a colored image but for the classifier to work we need a greyscale image
        #to convert
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #To store the faces
        #This will detect all the images in the current frame, and it will return the coordinates of the faces
        #Takes in image and some other parameter for accurate result
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        #In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
        for(x,y,w,h) in faces:
            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum+1

            print("saving dataset................")

            # Saving the image dataset, but only the face part, cropping the rest

            gray_img = gray[y:y+h,x:x+w]
            print("user name = {}".format(id))

            path =BASE_DIR+'/ml/dataset/'+str(id)+'_'+str(sampleNum)+'.jpg'


            cv2.imwrite(path,gray_img)
            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(250)

        #Showing the image in another window
        #Creates a window with window name "Face" and with the image img
        cv2.imshow("Face",img)
        #Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        #To get out of the loop
        if(sampleNum>35):
            break
    #releasing the cam
    cam.release()
    # destroying all the windows
    cv2.destroyAllWindows()

"""Now train the model
"""

def trainer():
    '''
        In trainer.py we have to get all the samples from the dataset folder,
        for the trainer to recognize which id number is for which face.
        for that we need to extract all the relative path
        i.e. dataset/user.1.1.jpg, dataset/user.1.2.jpg, dataset/user.1.3.jpg
        for this python has a library called os
    '''
    import os
    import numpy as np
    from PIL import Image

    #Creating a recognizer to train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #Path of the samples
    path =BASE_DIR+'/ml/dataset'

    # To get all the images, we need corresponing id
    def getImagesWithID(path):
        # create a list for the path for all the images that is available in the folder
        # from the path(dataset folder) this is listing all the directories and it is fetching the directories from each and every pictures
        # And putting them in 'f' and join method is appending the f(file name) to the path with the '/'
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #concatinate the path with the image name
        #print imagePaths

        # Now, we loop all the images and store that userid and the face with different image list
        faces = []
        Ids = []



        # Create a diction  for comvert name to integer

        dict={'a':1,
              'b':2,
              'c':3,
              'd':4,
              'e':5,
              'f':6,
              'g':7,
              'h':8,
              'i':9,
              'j':10,
              'k':11,
              'l':12,
              'm':13,
              'n':14,
              'o':15,
              'p':16,
              'q':17,
              'r':18,
              's':19,
              't':20,
              'u':21,
              'v':22,
              'w':23,
              'x':24,
              'y':25,
              'z':26}

        for imagePath in imagePaths:
            # First we have to open the image then we have to convert it into numpy array
            faceImg = Image.open(imagePath).convert('L') #convert it to grayscale
            # converting the PIL image to numpy array
            # @params takes image and convertion format
            faceNp = np.array(faceImg, 'uint8')
            # Now we need to get the user id, which we can get from the name of the picture
            # for this we have to slit the path() i.e dataset/user.1.7.jpg with path splitter and then get the second part only i.e. user.1.7.jpg
            # Then we split the second part with . splitter
            # Initially in string format so hance have to convert into int format

            print(imagePath)

            name =(os.path.split(imagePath)[-1].split('_')[0]) # -1 so that it will count from backwards and slipt the second index of the '.' Hence id
            # Images
            print(name)

            faces.append(faceNp)
            # Label
            """
             Need to make algo for generation ID
            
            """
            # pow=1
            # num=0
            # for i in range(len(name)-1):
            #     num+=dict[name[i]]*pow
            #     pow*=10

            ID = int(name)
            print("ID==================================== {}".format(ID))


            print(ID)
            Ids.append(ID)
            #print ID
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        return np.array(Ids), np.array(faces)

    # Fetching ids and faces
    ids, faces = getImagesWithID(path)

    #Training the recognizer
    # For that we need face samples and corresponding labels
    recognizer.train(faces, ids)

    # Save the recogzier state so that we can access it later
    recognizer.save(BASE_DIR+'/ml/recognizer/trainingData.yml')
    cv2.destroyAllWindows()

"""
   Now Detect
"""

from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.models import User




class PasswordlessAuthBackend(ModelBackend):
    """Log in to Django without providing a password.

    """
    def authenticate(self, username=None):
        try:
            return User.objects.get(username=username)
        except User.DoesNotExist:
            return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None

def detect():
    faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

    cam = cv2.VideoCapture(0)
    # creating recognizer
    rec = cv2.face.LBPHFaceRecognizer_create();
    # loading the training data
    rec.read(BASE_DIR+'/ml/recognizer/trainingData.yml')
    getId = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    userId = -1

    frame_no = 1
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        frame_no +=1
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)

            getId,conf = rec.predict(gray[y:y+h, x:x+w]) #This will predict the id of the face

            #print conf;
            if conf<35:
                userId = getId
                cv2.putText(img, "Detected",(x,y+h), font, 2, (0,255,0),2)
            else:
                cv2.putText(img, "Unknown",(x,y+h), font, 2, (0,0,255),2)

            # Printing that number below the face
            # @Prams cam image, id, location,font style, color, stroke

            if frame_no>100:
                break

        cv2.imshow("Face",img)
        if(cv2.waitKey(1) == ord('q')):
            break
        elif(userId != -1):
            return userId
            #cv2.waitKey(1000)
            cam.release()
            cv2.destroyAllWindows()
            return 1


    cam.release()
    cv2.destroyAllWindows()
    return 0


def login(request):

    # here need to authenticate with database username and password
    if request.method == 'POST':
        # now call according to html input name
        username = request.POST.get('username')
        password = request.POST.get('password')

        action_name = request.POST.get('action')

        print(action_name)

        if action_name == 'Login':
            #now authenticate
            user =PasswordlessAuthBackend.authenticate(request, username=username)

            if user is not None:
               print("yes valid=================================")
               dj_login(request, user)
               return redirect('home')
            else:
                messages.info(request,'Username of Password is incorrect!')
        else:
            print('face detection running')

            value= detect()
            username = str(value)
            print(username)
            user = PasswordlessAuthBackend.authenticate(request, username=username)

            if user is not None:
                print("yes valid=================================")
                print(value)
                """
                
                 Extract username and password according to face
                
                """

                dj_login(request, user)
                return redirect('home')





    return render(request,'login/login.html')






def registerPage(request):
    # request files is need for image
    form = CreateUserform(request.POST)

    folder = 'profiles/'
    if form.is_valid():
        print("yes for is valid===================")
        form.save()

        # retrieve user name for falsh message
        user = form.cleaned_data.get('username')

        # creating face verification dataset
        create_dataset(user)

        # Now training them.
        trainer()





        # myfile = request.FILES['image']
        # fs = FileSystemStorage(location=folder)
        # filename = fs.save(myfile.name,myfile)
        # print(filename)

        # for flash message
        messages.success(request, 'Registration done successfully for ' + user)

        # immediately redirect to login page
        return redirect('log')
    else:
        print("No form is not valid")
    #context = {'form':form}
    return render(request,'login/register.html', {'form':form})


def logoutUser(request):
    logout(request)
    return redirect('log')



"""Review process
"""

from ml.sent_analysis.analysis import predict
def prodRev(request):

    if request.method == "POST" and request.is_ajax():

        print("yes came------------")
        form = Rev(request.POST)

        if form.is_valid():
            form.save()
        # To access review
        review = request.POST.get('review')

        # predicting positive or negative review using ML sentiment analysis

        res = predict(review)
        print(res)

        if res =='pos':
            # For pos counter
            pos_review =Counter.objects.get(survey_wizard_type='survey_wizard_one')
            pos_review.pos_count = F('pos_count') + 1
            pos_review.save()



        elif res=='neg':
            # For neg counter
            neg_review = Counter.objects.get(survey_wizard_type='survey_wizard_one')
            neg_review.neg_count = F('neg_count') + 1
            neg_review.save()

        # For total counter
        tot_review = Counter.objects.get(survey_wizard_type='survey_wizard_one')
        tot_review.total_count = F('total_count') + 1
        tot_review.save()

        # Calculating the percentage

        tot_pos_rev=Counter.objects.values('pos_count')


        tot_neg_rev=Counter.objects.values('neg_count')


        tot_rev=Counter.objects.values('total_count')


        tot_rev=tot_rev[0]['total_count']
        tot_pos_rev = tot_pos_rev[0]['pos_count']
        tot_neg_rev = tot_neg_rev[0]['neg_count']

        print(tot_rev,tot_pos_rev,tot_neg_rev)

        pos_data = int((tot_pos_rev/tot_rev)*100)
        neg_data = int((tot_neg_rev/tot_rev)*100)
        print(pos_data,neg_data)


        return JsonResponse({"pos_data":pos_data,"neg_data":neg_data},status=200)

        # return JsonResponse({"success": True}, status=200)
    else:
        print("no success")
        return JsonResponse({"success": False}, status=400)









