import os
import json
from bs4 import BeautifulSoup
from urllib.request import urlopen

if __name__ == "__main__":
    # movie_list = 스파이더맨 파 프롬 홈, 어벤져스 엔드게임
    movie_list = ["https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=173123&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page=",
                  "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=136900&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page="]

    raw_folder = "raw_data"
    data_name = "data.json"

    # folder for download
    try:
        os.makedirs(os.path.join(raw_folder))
    except FileExistsError as e:
        pass

    # save crawled data
    with open(os.path.join(os.getcwd(), raw_folder, data_name), "w", encoding='utf-8') as f:
        for movie_idx, movie in enumerate(movie_list):
            count = 1
            while(True):
                html = urlopen(movie+str(count))
                bsObj = BeautifulSoup(html, "html.parser")

                # break condition : total_review/review_per_page < count
                total_review = int(bsObj.select("div > div.score_total > strong > em")[0].get_text().replace(",","")) / 10
                if(total_review < count):
                    break

                # get score, review & write to json file
                for child in bsObj.findAll("div", {"class":"score_result"})[0].ul.children:
                    if child != "\n":
                        score = child.findAll("div", {"class": "star_score"})[0].em.get_text()
                        review = child.findAll("div", {"class": "score_reple"})[0].p.get_text()

                        f.write(json.dumps({"score": score, "review": review}, ensure_ascii=False))
                        f.write("\n")

                # show processing
                if count % 100 == 0:
                    print("movie :", movie_idx, "\tcount :", count)

                count += 1
