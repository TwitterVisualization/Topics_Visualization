import React, { useState } from 'react';

import { addPrefetchExcludes } from 'react-static'
//import ForkMeOnGithub from 'fork-me-on-github';
import { Card, List } from 'antd';

//import './app.css'
import 'antd/dist/antd.css'; // or 'antd/dist/antd.less'

import data from "data.js"  // import data

// Any routes that start with 'dynamic' will be treated as non-static routes
addPrefetchExcludes(['dynamic'])

function App() {

  const [selectedMetaTopic, setSelectedMetaTopic] = useState(data[0])
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [isTopicLock, setIsTopicLock] = useState(false)

  const [overTopic, setOverTopic] = useState(null)
  const [overMeta, setOverMeta] = useState(null)

  const selectedColor = "#6cadde"
  const overColor = "#6cadde66"

  const clickOnMetaTopic = (metaTopic) => {
    setSelectedMetaTopic(metaTopic)
    setSelectedTopic(null)
  }
  const clickOnTopic = (topic) => {
    // unlock if you click on the same topic
    if (topic == selectedTopic && isTopicLock) {
      setIsTopicLock(false)
    } else {
      setIsTopicLock(true)
      setSelectedTopic(topic)
    }
  }

  return <div>
    {/*<ForkMeOnGithub repo="https://github.com/TwitterVisualization/Topics_Visualization/"/>*/}
    <ul>
      <li>
        <a href="https://github.com/TwitterVisualization/Topics_Visualization">Github repo</a>
      </li>
      <li>
        <a href="https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/TwitterVisualization/0fce80610be9b46486f9d5a24b39b994/raw/e31cb7f5b5b264ed6814554c2203cdb5e0070faf/template_projector_config.json">Embeddings visualization</a>
      </li>
      <li>
        <a href="/world_tweets.html">Tweets geolocation</a>
      </li>
    </ul>
    <div style={{ position: "fixed", top: "50%", left: "50%", transform: "translate(-50%, -50%)" }}>
      <div style={{ display: "flex", flexDirection: "row", spaceBetween: "10px" }} className="site-card-border-less-wrapper">
        <Card
          title={`Meta topic: ${selectedMetaTopic && selectedMetaTopic.name}`} bordered={true} style={{ margin: "10px", width: 300 }}>
          <List style={{ height: "600px", overflowY: "auto" }} >
            {data.map(metaTopic => <List.Item
              style={{ ...{ padding: "10px" }, ...(selectedMetaTopic == metaTopic) ? { backgroundColor: selectedColor } : (overMeta == metaTopic) ? { backgroundColor: overColor } : {} }}
              onMouseOver={() => {
                setOverMeta(metaTopic)
              }}
              onMouseOut={() => setOverMeta(null)}
              onClick={() => clickOnMetaTopic(metaTopic)} key={metaTopic.name}>{metaTopic.name}</List.Item>)}
          </List>
        </Card>
        <Card title={`Topics in meta topic ${selectedMetaTopic && selectedMetaTopic.name}`} bordered={true} style={{ margin: "10px", width: 300 }}>
          {(selectedMetaTopic) ? (
            <List style={{ height: "600px", overflowY: "auto" }}>
              {selectedMetaTopic.topics.map(topic => <List.Item
                style={{ ...{ padding: "10px" }, ...(selectedTopic == topic) ? { backgroundColor: selectedColor } : (overTopic == topic) ? { backgroundColor: overColor } : {} }}
                onMouseOver={() => {
                  if (!isTopicLock) {
                    setSelectedTopic(topic)
                  }
                  setOverTopic(topic)
                }}
                onMouseOut={() => setOverTopic(null)}
                onClick={() => clickOnTopic(topic)} key={topic.name}>
                {`Topic ${topic.name} (${Math.round(topic.count)})`}
              </List.Item>)}
            </List>
          ) : "Please select a Meta Topic"}
        </Card>

        <Card title={`Hashtags in topic ${selectedTopic && selectedTopic.name}`} bordered={true} style={{ margin: "10px", width: 300 }}>
          {(selectedTopic) ? (
            <List style={{ height: "600px", overflowY: "auto" }}>
              {selectedTopic.hashtags.map(hashtag => <List.Item key={hashtag.name}><a target="_blank" href={`https://twitter.com/search?q=${encodeURIComponent(hashtag.name)}&src=typed_query`}>{`${hashtag.name}`}</a> ({hashtag.count})</List.Item>)}
            </List>
          ) : "Please select a Topic"}
        </Card>

      </div>
    </div >
  </div>
}

export default App
